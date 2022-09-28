import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.models import FewShotModel


class ProtoRec(FewShotModel):

    def _forward(self, instance_embs, support_idx, query_idx):
        emb_dim = instance_embs.size(-1)
        # normalization
        # instance_embs /= torch.norm(instance_embs, dim=-1, p=2)[:, None]

        # organize support/query data
        support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (-1,)))
        query   = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape   + (-1,)))
    
        # get mean of the support
        proto = support.mean(dim=1) # Ntask x NK x d
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = np.prod(query_idx.shape[-2:])
        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        if self.args.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
            proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
            proto = proto.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)

            logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature
        else:
            proto = F.normalize(proto, dim=-1) # normalize for cosine distance
            query = query.view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)

            logits = torch.bmm(query, proto.permute([0,2,1])) / self.args.temperature
            logits = logits.view(-1, num_proto)

        return logits