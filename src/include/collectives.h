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

// Bine "2_blocks" AllGather (libpico allgather_bine_2_blocks): natural-order, no
// permutation. Each rank holds a contiguous (mod-wraparound) range of source
// blocks and DOUBLES it each step by exchanging 'mask' blocks with partner
// binePi(rank,step). mask = 1,2,4,...; the held range grows right or left
// depending on (step parity == rank parity).
//
// Concurrency model (PAT-style aggregation, the only safe form for a pairwise
// butterfly): a "message" packs up to postFreq block-slices into ONE FIFO slot at
// byte offsets 0,nelem,2*nelem,... with a SINGLE post. parallelFactor == postFreq,
// so the postFreq copies of a message run concurrently across worker groups while
// exactly one of them advances peer->step (no peer->step race). The op stream is a
// sequence of "waves" of exactly postFreq ops; each wave is single-direction (all
// sends, or all recvs, or the INIT) so it never deadlocks and never has an
// intra-wave output hazard. nelem == chunkCount (<= slot), never a large flat copy.
template <typename T>
class PatAGAlgorithm {
  size_t offset; // current chunk start within [0, count); advances by chunkCount
  size_t end;    // channelOffset + channelCount
  size_t count;  // elements per source rank (block stride in the output buffer)
  int chunkCount;
  int nelem;     // elements per block-slice this chunk (== chunkCount, clamped)
  int rank;
  int nranks;
  int nsteps;        // log2(nranks)
  int postFreq;      // block-slices packed per message == parallelFactor (#worker groups)

  // Schedule cursor. Layout per chunk: INIT wave (own-block copy + postFreq-1 skip
  // pads); then step=0..nsteps-1 (mask doubling), partner binePi(rank,step): a SEND
  // phase of the contiguous range [sendIndex,+mask) then a RECV phase of
  // [recvIndex,+mask). Each phase is emitted as ceil(mask/postFreq) message-waves;
  // the last wave is skip-padded up to postFreq.
  int phase;         // 0 = INIT, 1 = SEND, 2 = RECV
  int step;          // current step, 0 .. nsteps-1
  int mask;          // blocks exchanged this step (1,2,4,...)
  int myFirst;       // start block of our contiguous held range
  int sendIndex;     // start block of the range we send this step
  int recvIndex;     // start block of the range we receive this step
  int rangeStart;    // start block of the phase's range (sendIndex or recvIndex)
  int msgBase;       // block index within [0,mask) at the start of the current wave
  int pos;           // position within the current wave, 0 .. postFreq-1
  int m;             // real block-slices in the current message = min(postFreq, mask-msgBase)

  __device__ __host__ ssize_t min(ssize_t a, ssize_t b) {
    return (a < b) ? a : b;
  }

  __device__ __host__ int getNelem() {
    return min(chunkCount, end - offset);
  }

  // Compute send/recv ranges for step 'st' and begin its SEND phase.
  __device__ __host__ void startStep(int st) {
    sendIndex = myFirst;
    if (((st & 1) == (rank & 1))) {
      recvIndex = (sendIndex + mask) % nranks;          // grow range to the right
    } else {
      recvIndex = (sendIndex - mask + nranks) % nranks; // grow range to the left
      myFirst = recvIndex;
    }
    phase = 1; // SEND
    rangeStart = sendIndex;
    msgBase = 0;
    pos = 0;
    m = (int)min(postFreq, mask);
  }

public:
  __device__ __host__ PatAGAlgorithm(int slotBytes, int stepDepth, int maxParallelFactor, size_t offset, size_t end,
                                     size_t count, int chunkCount, int rank, int nranks)
    : offset(offset), end(end), count(count), chunkCount(chunkCount), rank(rank), nranks(nranks) {
    nsteps = log2Up(nranks);
    (void)stepDepth;
    // postFreq = how many nelem-slices fit one FIFO slot, capped by the worker-group
    // budget (maxParallelFactor). Power of two so it cleanly divides the worker pool.
    // slotBytes and chunkCount are identical on host (proxy) and device (kernel) so
    // postFreq matches. Small/mid messages -> small chunk -> large postFreq (high
    // concurrency, where the PAT gap is worst); large messages -> postFreq -> 1.
    int chunkBytes = chunkCount * (int)sizeof(T);
    int fit = chunkBytes > 0 ? slotBytes / chunkBytes : 1;
    if (fit < 1) fit = 1;
    postFreq = 1;
    while (postFreq * 2 <= fit && postFreq * 2 <= maxParallelFactor) postFreq *= 2;
    nelem = getNelem();
    phase = 0; // INIT
    pos = 0;
    step = 0;
    mask = 1;
    myFirst = rank;
  }

  __device__ __host__ int getParallelFactor() {
    return postFreq;
  }

  __device__ __host__ void getNextOp(struct ncclPatStep* ps) {
    ps->last = 0;
    ps->nelem = nelem;
    ps->recvDim = -1;
    ps->sendDim = -1;
    ps->recvOffset = 0;
    ps->sendOffset = 0;
    ps->stepOffset = 0;
    ps->postRecv = 0;
    ps->postSend = 0;
    ps->inpIx = 0;
    ps->outIx = 0;
    int skip = 0;

    if (phase == 0) {
      // INIT wave: pos 0 copies our own block; pos 1..postFreq-1 are skip pads so
      // the first SEND wave (which reads our output) lands in a later, barrier-
      // separated wave.
      if (pos == 0) {
        nelem = getNelem();
        ps->nelem = nelem;
        ps->inpIx = offset;
        ps->outIx = (size_t)rank * count + offset;
      } else {
        skip = 1;
      }
      if (nsteps == 0 && offset + chunkCount >= end) ps->last = (pos == postFreq - 1) ? 2 : 1;
      pos++;
      if (pos >= postFreq) {
        pos = 0;
        if (nsteps == 0) {
          if (offset + chunkCount < end) offset += chunkCount; // else: last==2 set above
        } else {
          step = 0;
          mask = 1;
          myFirst = rank;
          startStep(0);
        }
      }
    } else {
      // SEND (phase 1) or RECV (phase 2) message-wave. pos 0..m-1 are real slices
      // packed at byte offset pos*nelem of one FIFO slot; only pos==m-1 posts.
      int send = (phase == 1);
      int isFinalWave = (!send && step == nsteps - 1 && offset + chunkCount >= end && msgBase + postFreq >= mask);
      if (pos < m) {
        int b = (rangeStart + msgBase + pos) % nranks;
        ps->nelem = nelem;
        if (send) {
          ps->sendDim = step;
          ps->inpIx = (size_t)b * count + offset; // patCopy reads userOutput (forward-send)
          ps->sendOffset = pos * nelem;
          ps->postSend = (pos == m - 1) ? 1 : 0;
        } else {
          ps->recvDim = step;
          ps->outIx = (size_t)b * count + offset;
          ps->recvOffset = pos * nelem;
          ps->postRecv = (pos == m - 1) ? 1 : 0;
        }
      } else {
        skip = 1;
      }
      if (isFinalWave) ps->last = (pos == postFreq - 1) ? 2 : 1;
      pos++;
      if (pos >= postFreq) {           // wave complete
        pos = 0;
        msgBase += postFreq;
        if (msgBase < mask) {          // more messages in this phase
          m = (int)min(postFreq, mask - msgBase);
        } else if (send) {             // SEND phase done -> RECV phase
          phase = 2;
          rangeStart = recvIndex;
          msgBase = 0;
          m = (int)min(postFreq, mask);
        } else if (step < nsteps - 1) { // RECV done -> next step
          step++;
          mask <<= 1;
          startStep(step);
        } else if (offset + chunkCount < end) { // -> next chunk
          offset += chunkCount;
          phase = 0;
        }
        // else: final wave of the operation; last==2 was set above.
      }
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
#endif
