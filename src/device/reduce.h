/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"

namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
    ncclRing *ring = &ncclShmem.channel.ring;
    const int nranks = ncclShmem.comm.nRanks;
    const int rank = ncclShmem.comm.rank;
    const int prevRank = ring->userRanks[nranks-1];
    const int root = work->root;
    size_t chunkCount;
    size_t channelCount;
    size_t gridOffset;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), (size_t*)nullptr, &gridOffset, &channelCount, &chunkCount);
    size_t offset;
    int nelem;

    // Coverity reports that the callee treats &ring->next as an array.  However, due to the use of
    // FanSymmetric<1>, only the first element is ever accessed, so it's fine.
    // coverity[callee_ptr_arith:FALSE]
    Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0>
      prims(tid, nthreads, &ring->prev, &ring->next, work->sendbuff, work->recvbuff, work->redOpArg);

    if (prevRank == root) {
      for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
        offset = gridOffset + elemOffset;
        nelem = min(chunkCount, channelCount - elemOffset);
        prims.send(offset, nelem);
      }
    }
    else if (rank == root) {
      for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
        offset = gridOffset + elemOffset;
        nelem = min(chunkCount, channelCount - elemOffset);
        prims.recvReduceCopy(offset, offset, nelem, /*postOp=*/true);
      }
    }
    else {
      for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
        offset = gridOffset + elemOffset;
        nelem = min(chunkCount, channelCount - elemOffset);
        prims.recvReduceSend(offset, nelem);
      }
    }
  }
}

template <typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runBine(int tid, int nthreads, struct ncclDevWorkColl *work) {
    ncclBine *bine = &ncclShmem.channel.bine;
    if (bine->nSteps == 0 || bine->send == nullptr || bine->recv == nullptr)
    {
      // Fallback a ring
      if (tid == 0)
      {
        printf("Bine reduce not initialized on rank %d; falling back to ring\n", ncclShmem.comm.rank);
      }
      runRing<T, RedOp, Proto>(tid, nthreads, work);
      return;
    }

    // Partitioning parameters.
    ssize_t gridOffset, channelCount, chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T),
                    (ssize_t *)nullptr, &gridOffset, &channelCount, &chunkCount);

    T *inputBuf = (T *)work->sendbuff;  // per-rank local input
    T *outputBuf = (T *)work->recvbuff; // reduction destination

    const int rank = ncclShmem.comm.rank;
    const int nRanks = ncclShmem.comm.nRanks;
    const int root = work->root;
    const bool isRoot = (rank == root);
    const bool useDirect = (work->direct & (NCCL_P2P_READ | NCCL_P2P_WRITE)) != 0;
    ncclBineLogDirectAndAlias("reduce", work, tid);

    bool hasReduced = false;

    const size_t rootOffset = ((size_t)root * nRanks + rank) * bine->nSteps;

    // Walk steps in the reverse order of broadcast (distance-halving tree).
    for (int step = bine->nSteps - 1; step >= 0; --step)
    {
      const int stepIdx = rootOffset + step;
      const int sendPeer = bine->recv[stepIdx]; // parent to send to (se esiste)
      const int recvPeer = bine->send[stepIdx];  // child to receive from (se esiste)

      // Sanity: per passo, al massimo una azione è permessa.
      if (sendPeer >= 0 && recvPeer >= 0)
      {
        if (tid == 0)
        {
          printf("Bine reduce schedule invalid: rank %d step %d has both send and recv\n", rank, step);
        }
        // Fallback a ring
        runRing<T, RedOp, Proto>(tid, nthreads, work);
        return;
      }

      // === RECV-ONLY step: incorporate child into my partial (or create it) ===
      if (recvPeer >= 0)
      {
        int recvPeers[1] = {recvPeer};

        if (useDirect)
        {
          // Percorso diretto: caricamenti diretti device-to-device.
          if (hasReduced)
          {
            Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/1, Proto, 0>
                prim(tid, nthreads, recvPeers, /*sendPeers=*/nullptr, outputBuf, outputBuf, work->redOpArg,
                     /*group=*/0, /*connIndexRecv=*/0, /*connIndexSend=*/0, work);
            for (ssize_t e = 0; e < channelCount; e += chunkCount)
            {
              const ssize_t off = gridOffset + e;
              const int ne = (int)min(chunkCount, channelCount - e);
              // Riduci il chunk in arrivo nel parziale esistente in outputBuf.
              prim.directRecvReduceCopy(off, off, ne, /*postOp=*/isRoot);
            }
          }
          else
          {
            // Il primo incoming crea il parziale in outputBuf da inputBuf ⊕ incoming.
            Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/1, Proto, 0>
                prim(tid, nthreads, recvPeers, /*sendPeers=*/nullptr, inputBuf, outputBuf, work->redOpArg,
                     /*group=*/0, /*connIndexRecv=*/0, /*connIndexSend=*/0, work);
            for (ssize_t e = 0; e < channelCount; e += chunkCount)
            {
              const ssize_t off = gridOffset + e;
              const int ne = (int)min(chunkCount, channelCount - e);
              prim.directRecvReduceCopy(off, off, ne, /*postOp=*/isRoot);
            }
            hasReduced = true;
          }
        }
        else
        {
          // Non-direct path.
          if (hasReduced)
          {
            Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0>
                prim(tid, nthreads, recvPeers, /*sendPeers=*/nullptr, outputBuf, outputBuf, work->redOpArg,
                     /*group=*/0, /*connIndexRecv=*/0, /*connIndexSend=*/0, work);
            for (ssize_t e = 0; e < channelCount; e += chunkCount)
            {
              const ssize_t off = gridOffset + e;
              const int ne = (int)min(chunkCount, channelCount - e);
              prim.recvReduceCopy(off, off, ne, /*postOp=*/isRoot);
            }
          }
          else
          {
            Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0>
                prim(tid, nthreads, recvPeers, /*sendPeers=*/nullptr, inputBuf, outputBuf, work->redOpArg,
                     /*group=*/0, /*connIndexRecv=*/0, /*connIndexSend=*/0, work);
            for (ssize_t e = 0; e < channelCount; e += chunkCount)
            {
              const ssize_t off = gridOffset + e;
              const int ne = (int)min(chunkCount, channelCount - e);
              prim.recvReduceCopy(off, off, ne, /*postOp=*/isRoot);
            }
            hasReduced = true;
          }
        }
        continue; // esattamente una azione per passo
      }

      // === SEND-ONLY step: push upward ===
      if (sendPeer >= 0)
      {
        // Il root non dovrebbe mai inviare.
        if (isRoot)
        {
          if (tid == 0)
          {
            printf("Bine reduce schedule invalid: root rank %d has a send at step %d\n", rank, step);
          }
          runRing<T, RedOp, Proto>(tid, nthreads, work);
          return;
        }

        int sendPeers[1] = {sendPeer};

        if (useDirect)
        {
          Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/1, Proto, 0>
              prim(tid, nthreads, /*recvPeers=*/nullptr, sendPeers, inputBuf, outputBuf, work->redOpArg,
                   /*group=*/0, /*connIndexRecv=*/0, /*connIndexSend=*/0, work);
          for (ssize_t e = 0; e < channelCount; e += chunkCount)
          {
            const ssize_t off = gridOffset + e;
            const int ne = (int)min(chunkCount, channelCount - e);
            // Se ho il parziale, invio da output; altrimenti sono una foglia e invio il mio input.
            if (hasReduced)
              prim.directSendFromOutput(off, ne);
            else
              prim.directSend(off, off, ne);
          }
        }
        else
        {
          Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>
              prim(tid, nthreads, /*recvPeers=*/nullptr, sendPeers, inputBuf, outputBuf, work->redOpArg,
                   /*group=*/0, /*connIndexRecv=*/0, /*connIndexSend=*/0, work);
          for (ssize_t e = 0; e < channelCount; e += chunkCount)
          {
            const ssize_t off = gridOffset + e;
            const int ne = (int)min(chunkCount, channelCount - e);
            if (hasReduced)
              prim.sendFromOutput(off, ne);
            else
              prim.send(off, ne);
          }
        }
        continue;
      }
    }
  }
}

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<REDUCE_CHUNKSTEPS/REDUCE_SLICESTEPS, REDUCE_SLICESTEPS>;
    runRing<T, RedOp, Proto>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};

template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduce, T, RedOp, NCCL_ALGO_BINE, NCCL_PROTO_SIMPLE>
{
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl *work)
  {
    using Proto = ProtoSimple<REDUCE_CHUNKSTEPS / REDUCE_SLICESTEPS, REDUCE_SLICESTEPS>;
    runBine<T, RedOp, Proto>(tid, nthreads, work);
  }
};

template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduce, T, RedOp, NCCL_ALGO_BINE, NCCL_PROTO_LL>
{
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl *work)
  {
    runBine<T, RedOp, ProtoLL>(tid, nthreads, work);
  }
};

template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduce, T, RedOp, NCCL_ALGO_BINE, NCCL_PROTO_LL128>
{
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl *work)
  {
    runBine<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};
