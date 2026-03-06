/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "transport.h"
#include "bootstrap.h"
#include <sstream>

NCCL_PARAM(MultiSegmentRegister, "MULTI_SEGMENT_REGISTER", 1);

ncclResult_t ncclTransportRingConnect(struct ncclComm* comm) {
  struct ringConnInfo {
    bool useNetPXN;
    bool useGdr;
  };
  struct ringConnInfo* ringInfo = NULL;
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    comm->useGdr = true;
    comm->useNetPXN = false;
    for (int c = 0; c < comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->ring.prev, 1, &channel->ring.next, 0), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_RING], 0), ret, fail);
    if (ncclParamLocalRegister() || ncclParamGraphRegister()) {
      NCCLCHECK(ncclCalloc(&ringInfo, comm->nRanks));
      ringInfo[comm->rank].useGdr = comm->useGdr;
      ringInfo[comm->rank].useNetPXN = comm->useNetPXN;
      NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, ringInfo, sizeof(struct ringConnInfo)), ret, fail);
      for (int i = 0; i < comm->nRanks; ++i) {
        if (!ringInfo[i].useGdr) comm->useGdr = false;
        if (ringInfo[i].useNetPXN) comm->useNetPXN = true;
        if (comm->useGdr == false && comm->useNetPXN == true) break;
      }
    }
    INFO(NCCL_INIT, "Connected all rings, use ring PXN %d GDR %d", comm->useNetPXN, comm->useGdr);
  }
exit:
  free(ringInfo);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTransportTreeConnect(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    // Connect Trees
    for (int c = 0; c < comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_TREE_ARITY, channel->tree.down, 1, &channel->tree.up, 0), ret, fail);
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->tree.up, NCCL_MAX_TREE_ARITY, channel->tree.down, 0), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
    INFO(NCCL_INIT, "Connected all trees");
  }
exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTransportPatConnect(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    for (int mask=1; mask<comm->nRanks; mask<<=1) {
      int prevPeer = (comm->rank + mask) % comm->nRanks;
      int nextPeer = (comm->rank + comm->nRanks - mask) % comm->nRanks;
      for (int c = 0; c < comm->nChannels; c++) {
        NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &prevPeer, 1, &nextPeer, 0), ret, fail); // ReduceScatter
      }
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
      for (int c = 0; c < comm->nChannels; c++) {
        NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &nextPeer, 1, &prevPeer, 0), ret, fail); // AllGather
      }
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
    }
    INFO(NCCL_INIT, "Connected binomial trees");
  }
exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTransportBineConnect(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  if (comm == nullptr || comm->nRanks <= 1) return ret;

  const int nRanks = comm->nRanks;
  bool anySchedules = false;
  bool connectedAnyPeers = false;
  int connectedChannels = 0;
  struct ncclTopoGraph* setupGraph = nullptr;

  INFO(NCCL_INIT, "BINE connect rank %d cudaDev %d nRanks %d nChannels %d", comm->rank, comm->cudaDev, comm->nRanks, comm->nChannels);

  for (int c = 0; c < comm->nChannels; ++c) {
    struct ncclChannel* channel = comm->channels + c;
    const int steps = channel->bine.nSteps;

    INFO(NCCL_INIT,
         "Bine channel %d context steps %d host(send=%p recv=%p partner=%p index=%p order=%p) dev(send=%p recv=%p partner=%p index=%p order=%p)",
         c,
         steps,
         channel->bine.host.send,
         channel->bine.host.recv,
         channel->bine.host.partners,
         channel->bine.host.index,
         channel->bine.host.order,
         channel->bine.dev.send,
         channel->bine.dev.recv,
         channel->bine.dev.partners,
         channel->bine.dev.index,
         channel->bine.dev.order);

    if (steps == 0) {
      INFO(NCCL_INIT, "Bine channel %d has zero steps and will be skipped", c);
      continue;
    }

    if (steps > 0) {
      if (channel->bine.host.send == nullptr || channel->bine.host.recv == nullptr) {
        WARN("Bine send/recv tables are missing for channel %d", c);
        ret = ncclInternalError;
        goto fail;
      }

      if (channel->bine.host.partners == nullptr) {
        WARN("BINE partner table is missing for channel %d", c);
        ret = ncclInternalError;
        goto fail;
      }      
    }    

    auto isBineEnabledFor = [&](ncclFunc_t func) {
      for (int proto = 0; proto < NCCL_NUM_PROTOCOLS; ++proto) {
        if (comm->bandwidths[func][NCCL_ALGO_BINE][proto] > 0.0f) {
          return true;
        }
      }
      return false;
    };

    const bool hasStepTables = steps > 0 && channel->bine.host.send && channel->bine.host.recv;
    const bool hasDoublingTables = steps > 0 && channel->bine.host.partners;
    if (steps > 0 && !hasStepTables) {
      WARN("Bine send/recv tables are missing for channel %d", c);
      ret = ncclInternalError;
      goto fail;
    }
    if (steps > 0 && !hasDoublingTables) {
      WARN("Bine partner table is missing for channel %d", c);
      ret = ncclInternalError;
      goto fail;
    }

    const bool enableBroadcastPhase =
        hasStepTables &&
        (isBineEnabledFor(ncclFuncBroadcast) || isBineEnabledFor(ncclFuncAllReduce));
    const bool enableReducePhase =
        hasStepTables &&
        (isBineEnabledFor(ncclFuncReduce) || isBineEnabledFor(ncclFuncReduceScatter) ||
         isBineEnabledFor(ncclFuncAllReduce));
    const bool enableDoublingPhase =
        hasDoublingTables &&
        (isBineEnabledFor(ncclFuncReduceScatter) || isBineEnabledFor(ncclFuncAllGather) ||
         isBineEnabledFor(ncclFuncAllReduce));

    const bool channelHasEnabledPhase = enableBroadcastPhase || enableReducePhase || enableDoublingPhase;
    if (!channelHasEnabledPhase) {
      INFO(NCCL_INIT,
           "BINE channel %d has valid tables but all phases disabled (env bandwidths <= 0)",
           c);
      continue;
    }

    anySchedules = true;

    std::vector<int> sendPeers;
    std::vector<int> recvPeers;

    auto addPeer = [&](int peer, bool sendToPeer, bool recvFromPeer) {
      if (peer < 0 || peer == comm->rank) return;
      if (sendToPeer) sendPeers.push_back(peer);
      if (recvFromPeer) recvPeers.push_back(peer);
    };

    for (int root = 0; root < nRanks; ++root) {
      const int rank = comm->rank;
      const size_t rootOffset = ((size_t)root * nRanks + rank) * steps;

      if (steps > 0 && channel->bine.host.send && channel->bine.host.recv &&
          (enableBroadcastPhase || enableReducePhase)) {
        std::ostringstream rootLog;
        bool rootHasComm = false;
        for (int step = 0; step < steps; ++step) {
          const int stepIdx = rootOffset + step;
          const int sendPeer = channel->bine.host.send[stepIdx];
          const int recvPeer = channel->bine.host.recv[stepIdx];
          std::vector<std::string> annotations;
          if (enableBroadcastPhase) {
            std::vector<std::string> bcastEntries;
            if (sendPeer >= 0 && sendPeer != comm->rank) {
              bcastEntries.push_back(std::string("send->") + std::to_string(sendPeer));
            }
            if (recvPeer >= 0 && recvPeer != comm->rank) {
              bcastEntries.push_back(std::string("recv<-") + std::to_string(recvPeer));
            }
            if (!bcastEntries.empty()) {
              std::ostringstream entry;
              entry << "bcast:";
              for (size_t i = 0; i < bcastEntries.size(); ++i) {
                if (i > 0) entry << ",";
                entry << bcastEntries[i];
              }
              annotations.push_back(entry.str());
            }
          }
          if (enableReducePhase) {
            std::vector<std::string> reduceEntries;
            if (recvPeer >= 0 && recvPeer != comm->rank) {
              reduceEntries.push_back(std::string("send->") + std::to_string(recvPeer));
            }
            if (sendPeer >= 0 && sendPeer != comm->rank) {
              reduceEntries.push_back(std::string("recv<-") + std::to_string(sendPeer));
            }
            if (!reduceEntries.empty()) {
              std::ostringstream entry;
              entry << "reduce:";
              for (size_t i = 0; i < reduceEntries.size(); ++i) {
                if (i > 0) entry << ",";
                entry << reduceEntries[i];
              }
              annotations.push_back(entry.str());
            }
          }
          if (!annotations.empty()) {
            if (!rootHasComm) {
              rootLog << "BINE channel " << c << " root " << root << " steps:";
              rootHasComm = true;
            }
            rootLog << " H" << step << "[";
            for (size_t i = 0; i < annotations.size(); ++i) {
              if (i > 0) rootLog << ' ';
              rootLog << annotations[i];
            }
            rootLog << "]";
          }
          if (enableBroadcastPhase) {
            addPeer(sendPeer, /*sendToPeer=*/true, /*recvFromPeer=*/false);
            addPeer(recvPeer, /*sendToPeer=*/false, /*recvFromPeer=*/true);
          }
          if (enableReducePhase) {
            addPeer(sendPeer, /*sendToPeer=*/false, /*recvFromPeer=*/true);
            addPeer(recvPeer, /*sendToPeer=*/true, /*recvFromPeer=*/false);
          }
        }
        if (rootHasComm) {
          INFO(NCCL_INIT, "%s", rootLog.str().c_str());
        }
      }
    }

    if (enableDoublingPhase) {
      std::ostringstream doublingLog;
      bool hasDoublingComm = false;
      for (int step = 0; step < comm->bine.nSteps; ++step) {
        const int partner = channel->bine.host.partners[comm->rank * comm->bine.nSteps + step];
        addPeer(partner, /*sendToPeer=*/true, /*recvFromPeer=*/true);
        if (partner >= 0 && partner != comm->rank) {
          if (!hasDoublingComm) {
            doublingLog << "BINE channel " << c << " doubling steps:";
            hasDoublingComm = true;
          }
          doublingLog << " D" << step << "[partner=" << partner << "]";
        }
      }
      if (hasDoublingComm) {
        INFO(NCCL_INIT, "%s", doublingLog.str().c_str());
      }
    }

    auto dedup = [](std::vector<int>& peers) {
      std::sort(peers.begin(), peers.end());
      peers.erase(std::unique(peers.begin(), peers.end()), peers.end());
    };

    dedup(sendPeers);
    dedup(recvPeers);

    if (!sendPeers.empty() || !recvPeers.empty()) {
      std::ostringstream oss;
      oss << "BINE channel " << c << " sendPeers";
      for (int peer : sendPeers) oss << " " << peer;
      if (sendPeers.empty()) oss << " none";
      oss << " recvPeers";
      for (int peer : recvPeers) oss << " " << peer;
      if (recvPeers.empty()) oss << " none";
      INFO(NCCL_INIT, "%s", oss.str().c_str());

      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c,
          recvPeers.empty() ? 0 : (int)recvPeers.size(),
          recvPeers.empty() ? nullptr : recvPeers.data(),
          sendPeers.empty() ? 0 : (int)sendPeers.size(),
          sendPeers.empty() ? nullptr : sendPeers.data(),
          0), ret, fail);
      connectedAnyPeers = true;
      connectedChannels += 1;
    } else {
      INFO(NCCL_INIT, "BINE channel %d produced no distinct peers after dedup", c);
    }
  }

  if (!anySchedules) goto exit;

  if (!connectedAnyPeers) {
    WARN("BINE schedule tables produced no peer connections");
    ret = ncclInternalError;
    goto fail;
  }

  setupGraph = &comm->graphs[NCCL_ALGO_TREE];
  if (setupGraph->nChannels <= 0) {
    WARN("BINE connect rank %d selected topo graph for P2P setup without channels", comm->rank);
    ret = ncclInternalError;
    goto fail;
  }
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, setupGraph, 0), ret, fail);
  INFO(NCCL_INIT, "Connected BINE peers on %d channels", connectedChannels);

exit:
  if (!anySchedules) {
    INFO(NCCL_INIT,
         "BINE connect rank %d disabled: NCCL_ALGO tuning or bandwidths removed all phases",
         comm->rank);
  }
  return ret;
fail:
  goto exit;
}
