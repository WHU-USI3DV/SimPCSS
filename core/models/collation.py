import torch
import MinkowskiEngine as ME


class CollateFN:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, list_data):
        r"""
        Collation function for MinkowskiEngine.SparseTensor that creates batched
        coordinates given a list of dictionaries.
        """
        list_d = []
        list_idx = []
        # list_labels_down = {} # cz adds for generating low resolution labels 
        # for k,v in list_data[0]['labels_down'].items(): # 创造含有网络不同层级k,v对
        #     list_labels_down[k] = []

        for d in list_data:
            list_d.append((d["coordinates"], d["features"], d["labels"]))
            list_idx.append(d["idx"].view(-1, 1))
            # for k, v in d["labels_down"].items(): # 对网络不同层级的k,v对进行聚合
            #     list_labels_down[k].extend(v)    

        coordinates_batch, features_batch, labels_batch = ME.utils.SparseCollation(dtype=torch.float32,
                                                                                   device=self.device)(list_d)
        
        # for k, v in list_labels_down.items():
        #     list_labels_down[k] = torch.stack(list_labels_down[k], dim=0) # 转换数据格式，形成一维向量

        idx = torch.cat(list_idx, dim=0)
        return {"coordinates": coordinates_batch,
                "features": features_batch,
                "labels": labels_batch,
                "idx": idx,
                # "labels_down": list_labels_down.
                }
