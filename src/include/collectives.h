/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_COLLECTIVES_H_
#define NCCL_COLLECTIVES_H_

#include "nccl.h"
#include "nccl_tuner.h"
#include "device.h"
#include "compiler.h"
#include <stdlib.h> // abort() for the host-side op-list overflow guard (Bine AllGather)

#define NCCL_MAX_NET_SIZE (1024 * 1024 * 1024L) // Rather than send INT_MAX which is 2G-1, send a power of two.

// CHUNKSIZE must be a multiple of SLICESIZE
#define ALLREDUCE_SLICESTEPS (NCCL_STEPS / 4)
#define ALLREDUCE_CHUNKSTEPS (NCCL_STEPS / 2)
#define ALLGATHER_SLICESTEPS (NCCL_STEPS / 4)
#define ALLGATHER_CHUNKSTEPS (NCCL_STEPS / 2)
#define ALLTOALL_SLICESTEPS 1
#define ALLTOALL_CHUNKSTEPS 1
#define REDUCESCATTER_SLICESTEPS (NCCL_STEPS / 4)
#define REDUCESCATTER_CHUNKSTEPS (NCCL_STEPS / 2)
#define BROADCAST_SLICESTEPS 1
#define BROADCAST_CHUNKSTEPS 1
#define GATHER_SLICESTEPS 1
#define GATHER_CHUNKSTEPS 1
#define SCATTER_SLICESTEPS 1
#define SCATTER_CHUNKSTEPS 1
#define REDUCE_SLICESTEPS 1
#define REDUCE_CHUNKSTEPS 1
#define NCCL_MAX_SLICE_PER_CHUNK 2  // max value for CHUNKSTEPS/SLICESTEPS, must accord with above
#define NCCL_MAX_NET_SIZE (1024 * 1024 * 1024L) // Rather than send INT_MAX which is 2G-1, send a power of two.

const char* ncclFuncToString(ncclFunc_t op);
const char* ncclDevRedOpToString(ncclDevRedOp_t op);
const char* ncclDatatypeToString(ncclDataType_t type);
const char* ncclAlgoToString(int algo);
const char* ncclProtoToString(int proto);

inline int ncclTypeSize(ncclDataType_t type) {
  switch (type) {
  case ncclInt8:
  case ncclUint8:
  case ncclFloat8e4m3:
  case ncclFloat8e5m2:
    return 1;
  case ncclFloat16:
  case ncclBfloat16:
    return 2;
  case ncclInt32:
  case ncclUint32:
  case ncclFloat32:
    return 4;
  case ncclInt64:
  case ncclUint64:
  case ncclFloat64:
    return 8;
  default:
    return -1;
  }
}

#include <sys/types.h>

#define NCCL_MODE_NORMAL 0
#define NCCL_MODE_OFFSET 1
#define NCCL_MODE_PTR 2
struct ncclConnFifo {
  int mode;
  int offset;
  ssize_t size;
  void* ptr;
};

#include <stdio.h>

class RingAlgorithm {
protected:
  int refCount;
  int nRanks;
  int nStepsPerLoop;
  int chunkSteps;
  int sliceSteps;
  ssize_t sliceSize;
  ssize_t loopSize;
  ssize_t channelSize;
  uint8_t* sendbuff;
  uint8_t* recvbuff;
  void* sendMhandle;
  void* recvMhandle;
  void* srecvMhandle;

public:
  // this ring class is used by proxy thread to retrieve the send and recv buffer, size as well as corresponding
  // mem handle based on the current step of the proxy args. The derived ring algo class is AR, AG, and BC which
  // would be allocated during enqueue stage and copied to proxy side through shared memory. For each copy, we will
  // increase the refCount by incRefCount() since the same ring algo object can be referenced multiple times for send
  // and recv progress. After all steps are done, we decrease the refCount and only delete the ring object when
  // refCount == 0.
  virtual void getNextSendAddr(int curStep, uint8_t** sendbuffOut, size_t* sizeOut, void** mhandleOut) = 0;
  virtual void getNextRecvAddr(int curStep, uint8_t** recvbuffOut, size_t* sizeOut, void** mhandleOut) = 0;
  int incRefCount() {
    return (int)COMPILER_ATOMIC_ADD_FETCH(&refCount, 1, std::memory_order_relaxed);
  }
  int decRefCount() {
    return (int)COMPILER_ATOMIC_SUB_FETCH(&refCount, 1, std::memory_order_release);
  }
  RingAlgorithm() {
    refCount = 0;
  }
  virtual ~RingAlgorithm() {};
};

class RingARAlgorithm : public RingAlgorithm {
private:
  int ringIndex;
  int elemSize;
  ssize_t chunkSize;
  int slicePerChunk;

public:
  void getNextSendAddr(int curStep, uint8_t** sendbuffOut, size_t* sizeOut, void** mhandleOut) {
    int curLoop = curStep / nStepsPerLoop;
    int curLoopStage = (curStep % nStepsPerLoop) / chunkSteps;
    int chunkStage = curLoopStage % nRanks;
    int sliceStage = (curStep % chunkSteps) / sliceSteps;
    ssize_t elemOffset = curLoop * loopSize;
    ssize_t remSize = channelSize - elemOffset;
    ssize_t chunkOffset;
    ssize_t sliceOffset;
    ssize_t curSliceSize;
    ssize_t curChunkSize;
    ssize_t size;
    ssize_t nelem;
    int chunkId;

    if (remSize < loopSize) {
      curChunkSize = alignUp(divUp(remSize / elemSize, nRanks), 16 / elemSize) * elemSize;
    } else {
      curChunkSize = chunkSize;
    }
    chunkId = (ringIndex + nRanks - 1 - chunkStage) % nRanks;
    chunkOffset = chunkId * curChunkSize;
    nelem = std::min(remSize - chunkOffset, curChunkSize);
    curSliceSize = std::max(divUp(nelem / elemSize, 16 * slicePerChunk) * 16, sliceSize / elemSize / 32) * elemSize;
    sliceOffset = sliceStage * curSliceSize;

    if (nelem <= sliceOffset) {
      *sendbuffOut = sendbuff;
      *mhandleOut = sendMhandle;
    } else {
      if (curLoopStage == 0) {
        *sendbuffOut = sendbuff + elemOffset + chunkOffset + sliceOffset;
        *mhandleOut = sendMhandle;
      } else {
        *sendbuffOut = recvbuff + elemOffset + chunkOffset + sliceOffset;
        *mhandleOut = srecvMhandle;
      }
    }
    size = std::min(curSliceSize, nelem - sliceOffset);
    *sizeOut = size < 0 ? 0 : size;
    return;
  }

  void getNextRecvAddr(int curStep, uint8_t** recvbuffOut, size_t* sizeOut, void** mhandleOut) {
    int curLoop = curStep / nStepsPerLoop;
    int curLoopStage = ((curStep + chunkSteps) % nStepsPerLoop) / chunkSteps;
    int chunkStage = curLoopStage % nRanks;
    int sliceStage = (curStep % chunkSteps) / sliceSteps;
    ssize_t elemOffset = curLoop * loopSize;
    ssize_t remSize = channelSize - elemOffset;
    ssize_t chunkOffset;
    ssize_t sliceOffset;
    ssize_t curSliceSize;
    ssize_t curChunkSize;
    ssize_t size;
    ssize_t nelem;
    int chunkId;

    if (remSize < loopSize) {
      curChunkSize = alignUp(divUp(remSize / elemSize, nRanks), 16 / elemSize) * elemSize;
    } else {
      curChunkSize = chunkSize;
    }

    if (curLoopStage == 0) {
      chunkId = (ringIndex + 1) % nRanks;
    } else {
      chunkId = (ringIndex + nRanks - 1 - chunkStage) % nRanks;
    }

    chunkOffset = chunkId * curChunkSize;
    nelem = std::min(remSize - chunkOffset, curChunkSize);
    curSliceSize = std::max(divUp(nelem / elemSize, 16 * slicePerChunk) * 16, sliceSize / elemSize / 32) * elemSize;
    sliceOffset = sliceStage * curSliceSize;
    if (nelem <= sliceOffset) {
      *recvbuffOut = recvbuff;
    } else {
      *recvbuffOut = recvbuff + elemOffset + chunkOffset + sliceOffset;
    }
    if (sizeOut) {
      size = std::min(curSliceSize, nelem - sliceOffset);
      *sizeOut = size < 0 ? 0 : size;
    }
    *mhandleOut = recvMhandle;
    return;
  }

  RingARAlgorithm(const void* sendbuff, void* recvbuff, int nRanks, int ringIndex, int chunkSteps, int sliceSteps,
                  size_t chunkSize, size_t sliceSize, size_t gridOffset, size_t channelSize, int elemSize,
                  void* sendMhandle, void* recvMhandle, void* srecvMhandle) {
    this->ringIndex = ringIndex;
    this->nRanks = nRanks;
    this->nStepsPerLoop = 2 * (nRanks - 1) * chunkSteps;
    this->chunkSteps = chunkSteps;
    this->sliceSteps = sliceSteps;
    this->chunkSize = chunkSize;
    this->sliceSize = sliceSize;
    this->loopSize = nRanks * chunkSize;
    this->sendbuff = (uint8_t*)sendbuff + gridOffset;
    this->recvbuff = (uint8_t*)recvbuff + gridOffset;
    this->channelSize = channelSize;
    this->elemSize = elemSize;
    this->sendMhandle = sendMhandle;
    this->recvMhandle = recvMhandle;
    this->srecvMhandle = srecvMhandle;
    this->slicePerChunk = chunkSteps / sliceSteps;
  }
  ~RingARAlgorithm() {}
};

class RingAGAlgorithm : public RingAlgorithm {
private:
  int* ringRanks;
  int elemSize;
  ssize_t sendSize;
  int slicePerChunk;

public:
  void getNextSendAddr(int curStep, uint8_t** sendbuffOut, size_t* sizeOut, void** mhandleOut) {
    int curLoop = curStep / nStepsPerLoop;
    int chunkStage = (curStep % nStepsPerLoop) / chunkSteps;
    int sliceStage = (curStep % chunkSteps) / sliceSteps;
    ssize_t sliceOffset;
    ssize_t curSliceSize;
    ssize_t offset;
    ssize_t elemOffset = curLoop * loopSize;
    ssize_t chunkSize = std::min(loopSize, channelSize - elemOffset);
    ssize_t size;
    int rankDest;
    uint8_t* buff;
    void* mhandle;

    curSliceSize = std::max(divUp(chunkSize / elemSize, 16 * slicePerChunk) * 16, sliceSize / elemSize / 32) * elemSize;
    sliceOffset = sliceStage * curSliceSize;
    if (chunkStage == 0) {
      rankDest = ringRanks[0];
      offset = elemOffset + sliceOffset;
      buff = sendbuff + offset;
      mhandle = sendMhandle;
    } else {
      rankDest = ringRanks[nRanks - chunkStage];
      offset = elemOffset + rankDest * sendSize + sliceOffset;
      buff = recvbuff + offset;
      mhandle = srecvMhandle;
    }
    *sendbuffOut = buff;
    size = std::min(curSliceSize, channelSize - elemOffset - sliceOffset);
    *sizeOut = size < 0 ? 0 : size;
    *mhandleOut = mhandle;
    return;
  }

  void getNextRecvAddr(int curStep, uint8_t** recvbuffOut, size_t* sizeOut, void** mhandleOut) {
    int curLoop = curStep / nStepsPerLoop;
    int chunkStage = ((curStep + chunkSteps) % nStepsPerLoop) / chunkSteps;
    int sliceStage = (curStep % chunkSteps) / sliceSteps;
    ssize_t sliceOffset;
    ssize_t curSliceSize;
    ssize_t offset;
    ssize_t elemOffset = curLoop * loopSize;
    ssize_t chunkSize = std::min(loopSize, channelSize - elemOffset);
    ssize_t size;
    int rankDest;

    curSliceSize = std::max(divUp(chunkSize / elemSize, 16 * slicePerChunk) * 16, sliceSize / elemSize / 32) * elemSize;
    sliceOffset = sliceStage * curSliceSize;
    if (chunkStage == 0) {
      rankDest = ringRanks[1];
    } else {
      rankDest = ringRanks[nRanks - chunkStage];
    }
    offset = elemOffset + rankDest * sendSize + sliceOffset;
    *recvbuffOut = recvbuff + offset;
    if (sizeOut) {
      size = std::min(sliceSize, channelSize - elemOffset - sliceOffset);
      *sizeOut = size < 0 ? 0 : size;
    }
    *mhandleOut = recvMhandle;
  }

  RingAGAlgorithm(const void* sendbuff, void* recvbuff, int nRanks, int* ringRanks, int chunkSteps, int sliceSteps,
                  size_t chunkSize, size_t sliceSize, size_t gridOffset, size_t channelSize, int elemSize,
                  size_t sendSize, void* sendMhandle, void* recvMhandle, void* srecvMhandle) {
    this->ringRanks = ringRanks;
    this->nRanks = nRanks;
    this->nStepsPerLoop = (nRanks - 1) * chunkSteps;
    this->chunkSteps = chunkSteps;
    this->sliceSteps = sliceSteps;
    this->elemSize = elemSize;
    this->sliceSize = sliceSize;
    this->loopSize = chunkSize;
    this->sendSize = sendSize;
    this->channelSize = channelSize;
    this->sendbuff = (uint8_t*)sendbuff + gridOffset;
    this->recvbuff = (uint8_t*)recvbuff + gridOffset;
    this->sendMhandle = sendMhandle;
    this->recvMhandle = recvMhandle;
    this->srecvMhandle = srecvMhandle;
    this->slicePerChunk = chunkSteps / sliceSteps;
  }
  ~RingAGAlgorithm() {}
};

class RingBCAlgorithm : public RingAlgorithm {
private:
  int root;
  int rank;
  int nextRank;

public:
  void getNextSendAddr(int curStep, uint8_t** sendbuffOut, size_t* sizeOut, void** mhandleOut) {
    int curLoop = curStep / nStepsPerLoop;
    int sliceStage = (curStep % chunkSteps) / sliceSteps;
    ssize_t sliceOffset = sliceStage * sliceSize;
    ssize_t offset;
    ssize_t elemOffset = curLoop * loopSize;
    ssize_t size;
    uint8_t* buff;
    void* mhandle;

    offset = elemOffset + sliceOffset;
    if (offset >= channelSize) {
      buff = sendbuff;
      mhandle = sendMhandle;
    } else if (rank == root) {
      buff = sendbuff + offset;
      mhandle = sendMhandle;
    } else {
      buff = recvbuff + offset;
      mhandle = srecvMhandle;
    }
    *sendbuffOut = buff;
    size = std::min(sliceSize, channelSize - offset);
    *sizeOut = size < 0 ? 0 : size;
    *mhandleOut = mhandle;
    return;
  }

  void getNextRecvAddr(int curStep, uint8_t** recvbuffOut, size_t* sizeOut, void** mhandleOut) {
    int curLoop = curStep / nStepsPerLoop;
    int sliceStage = (curStep % chunkSteps) / sliceSteps;
    ssize_t sliceOffset = sliceStage * sliceSize;
    ssize_t offset;
    ssize_t elemOffset = curLoop * loopSize;
    ssize_t size;
    offset = elemOffset + sliceOffset;
    if (offset >= channelSize) {
      *recvbuffOut = recvbuff;
    } else {
      *recvbuffOut = recvbuff + offset;
    }
    if (sizeOut) {
      size = std::min(sliceSize, channelSize - offset);
      *sizeOut = size < 0 ? 0 : size;
    }
    *mhandleOut = recvMhandle;
    return;
  }

  RingBCAlgorithm(const void* sendbuff, void* recvbuff, int rank, int root, int nRanks, int* ringRanks, int chunkSteps,
                  int sliceSteps, size_t chunkSize, size_t sliceSize, size_t gridOffset, size_t channelSize,
                  void* sendMhandle, void* recvMhandle, void* srecvMhandle) {
    this->root = root;
    this->rank = rank;
    this->nextRank = ringRanks[1];
    this->nStepsPerLoop = chunkSteps;
    this->chunkSteps = chunkSteps;
    this->sliceSteps = sliceSteps;
    this->sliceSize = sliceSize;
    this->loopSize = chunkSize;
    this->channelSize = channelSize;
    this->sendbuff = (uint8_t*)sendbuff + gridOffset;
    this->recvbuff = (uint8_t*)recvbuff + gridOffset;
    this->sendMhandle = sendMhandle;
    this->recvMhandle = recvMhandle;
    this->srecvMhandle = srecvMhandle;
  }
  ~RingBCAlgorithm() {}
};

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#include <cuda/atomic>
#endif

// Need a power of two to ensure it divides by parallelFactor (which is also a power of two)
#define NCCL_PAT_NWORKERS 512

static constexpr int PatUsed = 0x1, PatSkipped = 0x2;

struct ncclPatStep {
  int recvDim, sendDim, recvOffset, sendOffset, stepOffset, postRecv, postSend, nelem, last, flags;
  size_t inpIx, outIx;
};

struct ncclPatPeer {
  uint64_t step;
  struct ncclConnInfo* conn;
  struct ncclConnFifo* connFifo;
  void* buff;
  uint64_t* headPtr;
  uint64_t* tailPtr;
  uint64_t stepCache;
  long long int accSize;
  int connStepSize;
};

#define NCCL_SHMEM_PAT_STEPS 32
struct ncclPatShmem {
  struct ncclPatStep patSteps[NCCL_SHMEM_PAT_STEPS];
  int parallelFactor;
  long long int localAccSize;
  struct ncclPatPeer sendDims[32]; // Should cover 2^32 ranks
  struct ncclPatPeer recvDims[32];
};

template <typename T>
class PatRSAlgorithm {
  size_t offset;
  size_t end;
  size_t count;
  int chunkCount;
  int nelem;
  int rank;
  int nranks;
  int nrPow2;
  int postFreq;
  int lastA;
  int parallelFactor;
  int aggFactor;
  int as; // aggregated steps
  int a; // step inside aggregated step
  int sendSkipped; // number of skipped steps during aggregation
  int stepOffset;
  int aggDelta;
  int scale;
  int phase;

  __device__ __host__ ssize_t min(ssize_t a, ssize_t b) {
    return (a < b) ? a : b;
  }

  __device__ __host__ int getNelem() {
    return min(chunkCount, end - offset);
  }

  __device__ __host__ int mirrorInvert(int i, int max) {
    int ret = 0;
    for (int mask = 1, imask = max / 2; mask < max; mask <<= 1, imask >>= 1) {
      if ((i & mask) == 0) ret += imask;
    }
    return ret;
  }

  __device__ __host__ int firstBitSet(int i, int max) {
    int ffs =
#ifdef __CUDA_ARCH__
      __ffs(i);
#else
      COMPILER_FFS(i);
#endif
    return ffs ? ffs - 1 : max;
  }

  __device__ __host__ void resetA() {
    a = 0;
    sendSkipped = stepOffset = 0;
    lastA = aggFactor;
    if (phase >= 2) lastA /= 2 * scale;
    if (phase == 4) lastA = 1;
  }

  __device__ __host__ void reset() {
    nelem = getNelem();
    phase = 0;
    scale = 1;
    as = aggDelta - 1;
    resetA();
  }

  __device__ __host__ int nBitsSet(int i) {
    int nbits =
#ifdef __CUDA_ARCH__
      __popc(i);
#else
      COMPILER_POPCOUNT32(i);
#endif
    return nbits;
  }

  // Return 1 when only upper bits are set. For example, if nrpow2==16 we'll return 1 for 8, 12, 14, 15.
  // A number being in the form of 1111000 implies that the complementary is 0000111 meaning it's a power of 2 minus 1.
  __device__ __host__ int newPeer(int i, int pow2) {
    // printf("New peer %d/%d -> %d\n", i, pow2, nBitsSet((i ^ (pow2-1)) + 1) == 1 ? 1 : 0);
    return nBitsSet((i ^ (pow2 - 1)) + 1) == 1 ? 1 : 0;
  }

public:
  __device__ __host__ PatRSAlgorithm(int stepSize, int stepDepth, int maxParallelFactor, size_t offset, size_t end,
                                     size_t count, int chunkCount, int rank, int nranks)
    : offset(offset), end(end), count(count), chunkCount(chunkCount), rank(rank), nranks(nranks) {
    parallelFactor = maxParallelFactor;
    aggDelta = nrPow2 = (1 << log2Up(nranks));

    aggFactor = 1;
    size_t channelSize = end - offset;
    while (stepSize / (channelSize * sizeof(T) * aggFactor) >= 2 && aggFactor < nranks / 2) {
      aggFactor *= 2;
      aggDelta /= 2;
    }
    postFreq = aggFactor;
    if (postFreq < parallelFactor) parallelFactor = postFreq;
    int d = stepDepth;
    while (d > 1 && aggFactor < nranks / 2) {
      d /= 2;
      aggFactor *= 2;
      aggDelta /= 2;
    }

    reset();
  }

  __device__ __host__ int getParallelFactor() {
    return parallelFactor;
  }

  __device__ __host__ void getNextOp(struct ncclPatStep* ps) {
    ps->last = 0;
    ps->nelem = nelem;
    ps->outIx = offset;
    ps->stepOffset = stepOffset;
    int skip = 0;
    if (a >= lastA) {
      skip = 1;
    } else if (phase == 0) {
      int s = mirrorInvert(a, lastA) * aggDelta + as;
      if (s >= nranks) skip = 1;
      int sendDataRank = (rank + s) % nranks;
      ps->inpIx = sendDataRank * count + offset;
      ps->recvDim = -1;
      ps->sendDim = 0;
      ps->outIx = 0;
      ps->recvOffset = -1;
      ps->sendOffset = (a % postFreq) * nelem;
      if (((a % postFreq) + 1 >= postFreq) || (a == lastA - 1)) {
        ps->postSend = 1;
      } else {
        ps->postSend = 0;
      }
      ps->postRecv = 0;
    } else if (phase == 1) {
      int s = mirrorInvert(a, lastA) * aggDelta + as;
      if (s >= nranks) skip = 1;
      ps->recvDim = firstBitSet(s, nrPow2);
      ps->sendOffset = (a % postFreq) * nelem;
      ps->recvOffset = (a % postFreq) * nelem;
      ps->postSend = 0;
      if (ps->recvDim == 0 && (((a % postFreq) + 1 >= postFreq) || (a == lastA - 1))) ps->postSend = 1;
      if (((a % postFreq) + 1 >= postFreq) || (a == lastA - 1)) {
        ps->postRecv = 1;
      } else {
        ps->postRecv = 0;
      }
      s -= (1 << ps->recvDim);
      int recvDataRank = (rank + nranks + s) % nranks;
      ps->inpIx = recvDataRank * count + offset;
      ps->sendDim = s ? firstBitSet(s, nrPow2) : -1;
      if (ps->sendDim == -1) {
        ps->sendOffset = -1;
      } else if (as - (1 << ps->recvDim) == 0) {
        if (newPeer(a, aggFactor)) {
          sendSkipped = a;
          ps->stepOffset = stepOffset = 0;
        }
        int foffset = a - sendSkipped;
        ps->sendOffset = (foffset % postFreq) * nelem;
      }
      int recvDim = ps->recvDim;
      if (s < nranks && skip) {
        ps->recvDim = -1;
        ps->recvOffset = -1;
        ps->postRecv = 0;
        skip = 0;
      }
      if (recvDim > 0 && (((a - sendSkipped) % postFreq) + 1 >= postFreq) && skip == 0) stepOffset++;
    } else if (phase == 2) {
      int s = (2 * mirrorInvert(a, lastA) + 1) * scale * aggDelta + 1;
      ps->postRecv = 0;
      if (s >= nranks) skip = 1;
      ps->recvDim = 0;
      ps->postSend = a == lastA - 1 ? 1 : 0;
      s -= 1;
      if (s < nranks && skip) {
        ps->recvDim = -1;
        ps->recvOffset = -1;
        skip = 0;
      } else if (!skip) {
        int foffset = a + aggFactor - aggFactor / scale;
        ps->postRecv |= ((foffset + 1) % postFreq) == 0 ? 1 : 0;
        ps->recvOffset = (foffset % postFreq) * nelem;
      }
      int recvDataRank = (rank + nranks + s) % nranks;
      ps->inpIx = recvDataRank * count + offset;
      ps->sendDim = s ? firstBitSet(s, nrPow2) : -1;
      int foffset = a;
      ps->postSend |= ((foffset + 1) % postFreq) == 0 ? 1 : 0;
      ps->sendOffset = (foffset % postFreq) * nelem;
    } else if (phase == 3) {
      int s = (2 * mirrorInvert(a, lastA) + 1) * scale * aggDelta;
      ps->postRecv = a == lastA - 1 ? 1 : 0;
      if (s >= nranks) skip = 1;
      ps->recvDim = firstBitSet(s, nrPow2);
      ps->postSend = 0;
      s -= (1 << ps->recvDim);
      int foffset = a;
      ps->postRecv |= (foffset + 1) % postFreq == 0 ? 1 : 0;
      ps->recvOffset = (foffset % postFreq) * nelem;
      int recvDataRank = (rank + nranks + s) % nranks;
      ps->inpIx = recvDataRank * count + offset;
      ps->sendDim = s ? firstBitSet(s, nrPow2) : -1;
      if (s < nranks && skip) {
        ps->recvDim = -1;
        ps->recvOffset = -1;
        ps->postRecv = 0;
        skip = 0;
      }
      if (newPeer(a, aggFactor / (2 * scale))) {
        sendSkipped = a;
        ps->stepOffset = stepOffset = 0;
      }
      foffset = a - sendSkipped;
      if ((foffset % postFreq) + 1 >= postFreq && skip == 0) stepOffset++;
      ps->sendOffset = ps->sendDim >= 0 ? (foffset % postFreq) * nelem : -1;
    } else if (phase == 4) {
      ps->recvDim = 0;
      ps->sendDim = -1;
      ps->inpIx = rank * count + offset;
      ps->recvOffset = ((aggFactor - 1) % postFreq) * nelem;
      ps->sendOffset = -1;
      ps->postRecv = 1;
      ps->postSend = 0;
      offset += chunkCount;
    }
    a++;
    if (a >= lastA && a >= parallelFactor) {
      int p = phase;
      if (p == 1) as--;
      if (p == 3) scale *= 2;
      phase = p == 0 ? as == 1 ? (aggFactor > 1 ? 2 : 4) : 1 :
              p == 1 ? as % 2 == 1 ? 0 : 1 :
              p == 2 ? 3 :
              p == 3 ? scale < aggFactor ? 2 : 4 :
                       5;
      if (p == 4) {
        if (offset >= end) {
          ps->last = 2;
        } else {
          reset();
        }
      } else {
        resetA();
      }
    } else if (phase == 4 && offset >= end) {
      ps->last = 1;
    }
    int flags = PatUsed | (skip ? PatSkipped : 0);
#if __CUDA_ARCH__ >= 600
    cuda::atomic_ref<int, cuda::thread_scope_block> a(ps->flags);
    a.store(flags, cuda::memory_order_release);
#else
    ps->flags = flags;
#endif
  }
};

// ---------------------------------------------------------------------------
// Bine (binomial-negabinary) AllGather schedule.
//
// Instead of PAT's binomial trees over ±2^k peers, Bine uses a *pairwise*
// (recursive-doubling-like) schedule: at each of the log2(nranks) rounds a rank
// exchanges with a single partner, chosen by pi() below, and sends-to ==
// receives-from the same partner.
//
//   rho_s = sum_{i=0..s} (-2)^i  ->  {1,-1,3,-5,11,-21,43,...}    (always odd)
//   pi(rank, step) = (rank +/- rho_step) mod nranks  (+ for even rank, - for odd)
//
// Because rho_s is odd the partner always has opposite parity, which makes pi()
// an involution: pi(pi(r,s),s) == r. That is what makes the schedule pairwise.
// See HLC-Lab/pico libpico_allgather.c (allgather_bine_block_by_block) and the
// Bine paper (arXiv 2508.17311). Helpers are __device__ __host__ so the same
// code drives the device kernel and the host proxy step-counter.
__device__ __host__ inline int bineRho(int step) {
  int r = 1, p = 1;
  for (int i = 1; i <= step; i++) { p *= -2; r += p; }
  return r;
}

__device__ __host__ inline int binePi(int rank, int step, int nranks) {
  int rho = bineRho(step);
  int dest = ((rank & 1) == 0) ? (rank + rho) % nranks : (rank - rho) % nranks;
  if (dest < 0) dest += nranks;
  return dest;
}

// Bine AllGather over negabinary edges. TWO SCHEDULES, one op-list class, picked at
// construction from the chunk size (see postFreq / useButterfly in the constructor):
//
//   RELAY (large messages): each rank's block is broadcast down a tree of negabinary
//     neighbours binePi(rank,k); all n source-trees run together, so a rank talks to up
//     to nsteps DIFFERENT peers (the multi-peer width PAT has, the pairwise butterfly
//     lacks) -- this is what beats PAT at large sizes. A relayed block is RECEIVED AND
//     FORWARDED in ONE fused op (recvDim>=0 && sendDim>=0): patCopy reads the recv-FIFO
//     once and writes BOTH the output and the child's send-FIFO. One block per FIFO slot.
//   BUTTERFLY (small/mid messages): pico's allgather_bine_block_by_block -- log2(n) rounds
//     of pairwise exchange with partner binePi(rank,t), packing postFreq blocks into each
//     FIFO slot so one network post covers many blocks. Latency-optimal where the relay's
//     one-post-per-block cost dominates. Selected when THIS CHANNEL's per-rank bytes
//     ((end-offset)*sizeof(T)) <= BINE_BUTTERFLY_MAX_BYTES, provided a slot can hold
//     enough of the largest round to be deadlock-free (postFreq >= divUp(nranks/2,
//     NCCL_STEPS)); otherwise the relay runs. The per-channel gate is required for
//     host(proxy)/device(kernel) mode agreement -- see the constructor.
//
// Op kinds (both modes): INIT (own block input->output), FUSED (recv+forward, relay only),
// SEND-only (forward a held block, reads output), RECV-only (leaf/plain receive). Each op
// carries opSlotPos (byte offset within its shared slot = slotPos*nelem) and opPost (which
// of postSend/postRecv fire); relay uses slotPos 0 and posts every op. parallelFactor=1 in
// both modes: concurrency is cross-peer via the FIFO, not worker groups. sendDim/recvDim =
// step index k -> connection peer binePi(rank,k) (generic.cc connects these; send-peer ==
// recv-peer per dim), so BOTH schedules reuse the same connections and proxy step-counter.
//
// The RELAY emission order matters for deadlock-freedom under bounded FIFOs:
//
// EMISSION ORDER = SKEWED SOURCE ORDER: blocks are emitted in ascending
// key(s) = s + BINE_SKEW_LAMBDA * depth(s), ties by s, where depth(s) is this rank's
// depth in block s's broadcast tree. This one-parameter family interpolates between two
// orders that both fail:
//   lambda = 0 (plain source order): deadlock-free but SLOW -- every rank consumes each
//     block at the same wavefront instant its parent produces it, so every fused op
//     stalls a full network hop, serialized (measured 0.1x PAT at 64 nodes on Leonardo).
//   lambda = inf (depth order): pipelines beautifully (blocks are consumed in arrival
//     order) but its per-connection bursts equal binomial(log2(n)-1, d), which exceeds
//     the NCCL_STEPS=8 FIFO depth at nranks >= 64 -> mutual dim-0 deadlock.
// Intermediate lambda emits a block's ops ~lambda positions later per tree-hop, so data
// arrives ~one hop before it is consumed (throughput) while per-connection send/recv
// skew stays bounded (liveness). Per-connection order consistency is preserved for ANY
// lambda: on the connection r->p the receiver's depth is the sender's +1, so p's keys
// are r's shifted by exactly +lambda -- same relative order on both endpoints.
// lambda = 6 was chosen with the timed + liveness simulators (bench_bine/timed_sim.py,
// bench_bine/verify_schedule.py): it is deadlock-free down to FIFO depth 6 (two slots of
// margin below the real 8) for all po2 nranks <= 256, and within ~5% of the best
// throughput in the model; lambda >= 12 deadlocks at n >= 64, lambda = 8 has zero
// margin. INIT is emitted at key == rank (depth 0); its own-block forwards immediately
// follow it, and every SEND-only op still comes after the op that gathered its block.
//
// The op list (kind/rdim/sdim/src) depends only on rank/nranks -> identical on
// host(proxy, T=char) and device(kernel, T); byte offsets scale by sizeof(T). Built once
// at construction (a single O(n log n) pass over the negabinary trees on the compute
// thread), replayed per chunk. Sized for po2 nRanks <= 256 (tuning.cc restricts use).
// Emission-order skew (list positions per tree-hop). MUST stay in lockstep with the
// mirror in bench_bine/verify_schedule.py; changing it requires re-running the phase2
// gate there (liveness) and bench_bine/timed_sim.py (throughput model).
#define BINE_SKEW_LAMBDA 6

// Butterfly (packed pairwise) is used when THIS CHANNEL's per-rank byte count
// ((end-offset)*sizeof(T)) is at or below this threshold; above it the multi-peer relay
// wins. Gated on the per-channel figure because that is the only size signal that is
// consistent between the host proxy and the device kernel (see mode selection in the
// constructor); a full-per-rank or postFreq gate either hangs or regresses. First-cut
// crossover; refine with the BINE_FORCE_RELAY / BINE_FORCE_BUTTERFLY builds. Keep in
// lockstep with the mirror in bench_bine/verify_schedule.py.
#define BINE_BUTTERFLY_MAX_BYTES (128 * 1024)

template <typename T>
class PatAGAlgorithm {
  static const int RMAXOPS = 520; // >= ~1.5*nRanks for nRanks <= 256
  size_t offset; // current chunk start within [0, count); advances by chunkCount
  size_t end;    // channelOffset + channelCount
  size_t count;  // elements per source rank (block stride in the output buffer)
  int chunkCount;
  int nelem;     // elements per block-slice this chunk (== chunkCount, clamped)
  int rank;
  int nranks;
  int nsteps;        // log2(nranks)

  // Precomputed op list (structure only; offsets computed at emit time).
  short opSrc[RMAXOPS];          // source block (data rank)
  signed char opRdim[RMAXOPS];   // recv dim (-1 if none) -> recv from binePi(rank,opRdim)
  signed char opSdim[RMAXOPS];   // send dim (-1 if none) -> send to  binePi(rank,opSdim)
  unsigned char opKind[RMAXOPS]; // 0 INIT, 1 SEND-only, 2 RECV-only, 3 FUSED (recv+send)
  unsigned char opSlotPos[RMAXOPS]; // block's position within its FIFO slot (0..postFreq-1)
  unsigned char opPost[RMAXOPS];    // bit0 = postSend, bit1 = postRecv on this op
  int nOps;
  int ip;                        // current op index within the chunk

  __device__ __host__ ssize_t imin(ssize_t a, ssize_t b) { return (a < b) ? a : b; }
  __device__ __host__ int getNelem() { return (int)imin(chunkCount, end - offset); }

  // Ascending insertion sort (m small: <= nranks/2). Butterfly send/recv runs are
  // enumerated in the same order on both endpoints, so sorting is only to keep the
  // C++ list byte-identical to the Python mirror; either order would be consistent.
  __device__ __host__ void sortAsc(int* a, int m) {
    for (int i = 1; i < m; i++) { int v = a[i], j = i - 1; while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; j--; } a[j + 1] = v; }
  }

  // Enumerate get_indexes(start,st) in DFS pre-order into out[]; returns count.
  // If dep != nullptr, dep[i] = tree depth of out[i] below 'start' (e0 has depth 1);
  // when start == rank this is exactly rank's depth in out[i]'s broadcast tree.
  __device__ __host__ int getIdx(int start, int st, int* out, int* dep = nullptr) {
    if (st >= nsteps) return 0;
    int e0 = binePi(start, st, nranks); int no = 0;
    if (dep) dep[no] = 1;
    out[no++] = e0;
    int node[32], sval[32], dstk[32], top = 0;
    node[0] = e0; sval[0] = st + 1; dstk[0] = 1; top = 1;
    while (top > 0) {
      int y = node[top - 1], s = sval[top - 1], d = dstk[top - 1];
      if (s >= nsteps) { top--; continue; }
      sval[top - 1] = s + 1;
      int c = binePi(y, s, nranks);
      if (dep) dep[no] = d + 1;
      out[no++] = c; node[top] = c; sval[top] = s + 1; dstk[top] = d + 1; top++;
    }
    return no;
  }
  // slotPos = block's offset within its shared FIFO slot; postBits bit0=postSend,
  // bit1=postRecv. Relay defaults (0, both bits): one block per slot, post every op.
  __device__ __host__ void push(int kind, int rd, int sd, int s, int slotPos = 0, int postBits = 3) {
    if (nOps >= RMAXOPS) {
      // Must never happen for guarded po2 nRanks <= 256 (relay worst case 2n-1 = 511,
      // butterfly 2n-1 = 511, both < 520); fail loudly rather than silently drop an op.
#ifdef __CUDA_ARCH__
      __trap();
#else
      abort();
#endif
    }
    opKind[nOps] = (unsigned char)kind; opRdim[nOps] = (signed char)rd; opSdim[nOps] = (signed char)sd;
    opSrc[nOps] = (short)s; opSlotPos[nOps] = (unsigned char)slotPos; opPost[nOps] = (unsigned char)postBits;
    nOps++;
  }

public:
  __device__ __host__ PatAGAlgorithm(int slotBytes, int stepDepth, int maxParallelFactor, size_t offset, size_t end,
                                     size_t count, int chunkCount, int rank, int nranks)
    : offset(offset), end(end), count(count), chunkCount(chunkCount), rank(rank), nranks(nranks) {
    (void)stepDepth; (void)maxParallelFactor;
    nsteps = log2Up(nranks);
    nelem = getNelem();

    // postFreq = blocks packed into one FIFO slot / one network post. Computed
    // identically on host (T=char, chunkCount in bytes) and device (chunkCount in
    // elements, *sizeof(T)) since both equal the chunk byte size, keeping the two
    // op lists in lockstep. Clamp to [1, nranks/2] (n/2 = largest butterfly round).
    int chunkBytes = chunkCount * (int)sizeof(T);
    int postFreq = (chunkBytes > 0) ? (slotBytes / chunkBytes) : 1;
    if (postFreq < 1) postFreq = 1;
    if (postFreq > nranks / 2) postFreq = nranks / 2;
    int minPost = (nranks / 2 + NCCL_STEPS - 1) / NCCL_STEPS; // divUp(nranks/2, NCCL_STEPS)
    // Mode selection. Butterfly (packed pairwise, few posts) wins the small/mid
    // latency-bound regime; the skewed relay (multi-peer width) wins large.
    //
    // GATE ON THIS CHANNEL'S PER-RANK BYTES = (end-offset)*sizeof(T). This quantity is
    // the ONLY size signal that is BOTH (a) host/device CONSISTENT and (b) a sensible
    // regime indicator. It must be host/device consistent or the proxy (host) and kernel
    // (device) pick different modes -> different op lists -> different per-dim step
    // counts -> NETWORK HANG. Two traps learned the hard way:
    //   * postFreq alone (= slotBytes/chunkBytes): consistent, but channel-SENSITIVE
    //     (more channels shrink the chunk, raise postFreq) -> butterfly wrongly chosen
    //     for large messages under many channels -> 1 GB 10.2->6.0 GB/s at 16 channels.
    //   * count*sizeof(T) (full per-rank): channel-invariant, but 'count' is the FULL
    //     per-rank size on the device and the PER-CHANNEL size on the host proxy
    //     (proxy.cc passes size=nbytes/nRanks as count) -> the two sides disagree above
    //     2 channels -> DEADLOCK.
    // (end-offset) sidesteps both: on the device it is channelCount, on the host it is
    // 'size'; the chunk loop already relies on these being equal, so the mode decision
    // inherits that guarantee. It scales down with channel count in the RIGHT direction
    // (less data per channel -> more latency-bound -> butterfly). postFreq >= minPost is
    // kept as the packing-safety floor (below it the butterfly deadlocks).
    // BINE_BUTTERFLY_MAX_BYTES is per-channel-per-rank; tune with the BINE_FORCE_* builds.
    size_t perChanBytes = (size_t)(end - offset) * sizeof(T);
#if defined(BINE_FORCE_RELAY)
    bool useButterfly = false;
#elif defined(BINE_FORCE_BUTTERFLY)
    bool useButterfly = (postFreq >= minPost);   // benchmarking: butterfly wherever it is SAFE
#else
    bool useButterfly = (perChanBytes <= BINE_BUTTERFLY_MAX_BYTES) && (postFreq >= minPost);
#endif

    nOps = 0;
    if (useButterfly) {
      // ---- Butterfly: pico allgather_bine_block_by_block (packed) ----
      // Rounds t = nsteps-1..0, partner p = binePi(rank,t) on connection dim t:
      // send the blocks p needs (getIdx(p,t), already gathered), then receive the
      // blocks r needs (getIdx(r,t)). Per-connection order matches because both
      // endpoints enumerate the same sorted set. postFreq blocks share a slot.
      int sbuf[260], rbuf[260];
      push(0, -1, -1, rank);                   // INIT: own block input -> output
      for (int t = nsteps - 1; t >= 0; t--) {
        int partner = binePi(rank, t, nranks);
        int ns = getIdx(partner, t, sbuf); sortAsc(sbuf, ns);   // blocks r sends to partner
        int nr = getIdx(rank, t, rbuf);    sortAsc(rbuf, nr);   // blocks r receives from partner
        for (int j = 0; j < ns; j++) {
          int post = (j % postFreq == postFreq - 1) || (j == ns - 1);
          push(1, -1, t, sbuf[j], j % postFreq, post ? 1 : 0);  // SEND-only (bit0 = postSend)
        }
        for (int j = 0; j < nr; j++) {
          int post = (j % postFreq == postFreq - 1) || (j == nr - 1);
          push(2, t, -1, rbuf[j], j % postFreq, post ? 2 : 0);  // RECV-only (bit1 = postRecv)
        }
      }
    } else {
      // ---- Skewed source-order relay (see class comment) ----
      // Precompute per source block s: krecv[s] = the dim r receives s on (-1 if
      // s==r), dof[s] = r's depth in s's broadcast tree (free from the same DFS),
      // cmask[s] = bitmask of child dims r forwards s to. From the negabinary trees:
      // getIdx(rank,k) is the set r receives via dim k; getIdx(binePi(rank,k),k) is
      // the set r forwards to its child on dim k.
      signed char krecv[260]; unsigned char cmask[260]; short dof[260];
      for (int s = 0; s < nranks; s++) { krecv[s] = -1; cmask[s] = 0; dof[s] = 0; }
      int set[260], dep[260];
      for (int k = 0; k < nsteps; k++) {
        int c = getIdx(rank, k, set, dep);
        for (int i = 0; i < c; i++) { krecv[set[i]] = (signed char)k; dof[set[i]] = (short)dep[i]; }
        c = getIdx(binePi(rank, k, nranks), k, set);
        for (int i = 0; i < c; i++) cmask[set[i]] |= (1u << k);
      }
      // Emit blocks by ascending key(s) = s + BINE_SKEW_LAMBDA*dof[s], ties by smaller
      // s. O(n^2) selection scan; trivial next to the data movement.
      unsigned char emitted[260];
      for (int s = 0; s < nranks; s++) emitted[s] = 0;
      for (int e = 0; e < nranks; e++) {
        int s = -1, bestKey = 0x7fffffff;
        for (int x = 0; x < nranks; x++) {
          if (emitted[x]) continue;
          int key = x + BINE_SKEW_LAMBDA * (int)dof[x];
          if (key < bestKey) { bestKey = key; s = x; }     // strict '<': equal keys -> smaller x
        }
        emitted[s] = 1;
        if (s == rank) {
          push(0, -1, -1, rank);                 // INIT: own block input -> output
          for (int k = 0; k < nsteps; k++) if (cmask[s] & (1u << k)) push(1, -1, k, s); // forward own block to every child
        } else {
          int first = -1;
          for (int k = 0; k < nsteps; k++) if (cmask[s] & (1u << k)) { first = k; break; }
          if (first >= 0) {
            push(3, krecv[s], first, s);         // FUSED: recv s from parent + forward to first child
            for (int k = first + 1; k < nsteps; k++) if (cmask[s] & (1u << k)) push(1, -1, k, s); // extra forwards (read output)
          } else {
            push(2, krecv[s], -1, s);            // leaf: receive only
          }
        }
      }
    }
    ip = 0;
  }

  __device__ __host__ int getParallelFactor() { return 1; }

  __device__ __host__ void getNextOp(struct ncclPatStep* ps) {
    ps->last = 0; ps->recvDim = -1; ps->sendDim = -1; ps->recvOffset = 0; ps->sendOffset = 0;
    ps->stepOffset = 0; ps->postRecv = 0; ps->postSend = 0; ps->inpIx = 0; ps->outIx = 0;
    nelem = getNelem(); ps->nelem = nelem;
    int K = opKind[ip], rd = opRdim[ip], sd = opSdim[ip], s = opSrc[ip];
    int slotPos = opSlotPos[ip], postBits = opPost[ip];
    if (K == 0) {                                              // INIT: own block input -> output[rank]
      ps->inpIx = offset; ps->outIx = (size_t)rank * count + offset;
    } else {
      // FUSED sets both; SEND-only sets send; RECV-only sets recv. postFreq blocks share a
      // slot: byte offset slotPos*nelem, and post only on the pack's last block (opPost).
      // Relay: slotPos 0, opPost = both bits -> one block per slot, post every op (unchanged).
      if (rd >= 0) { ps->recvDim = rd; ps->outIx = (size_t)s * count + offset; ps->recvOffset = slotPos * nelem; if (postBits & 2) ps->postRecv = 1; }
      if (sd >= 0) { ps->sendDim = sd; ps->inpIx = (size_t)s * count + offset; ps->sendOffset = slotPos * nelem; if (postBits & 1) ps->postSend = 1; }
    }
    ip++;
    if (ip >= nOps) {
      if (offset + chunkCount >= end) ps->last = 2;   // final op of the operation
      else { offset += chunkCount; ip = 0; }          // next chunk: replay op list
    }
    int flags = PatUsed;
#if __CUDA_ARCH__ >= 600
    cuda::atomic_ref<int, cuda::thread_scope_block> a(ps->flags);
    a.store(flags, cuda::memory_order_release);
#else
    ps->flags = flags;
#endif
  }
};
#endif
