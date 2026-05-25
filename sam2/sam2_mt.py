from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.transforms import SAM2Transforms
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch

class MDMTSAM2Predictor(nn.Module):
    def __init__(
        self,
        sam_model: SAM2Base,
        domain2task2slice,
        **kwargs,
    ):
        super().__init__()
        self.model = sam_model
        
        # self.model = sam_model
        # print(self.model.device)
        # self.model = sam_model
        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=0,
        )
        self.device = self.model.device

        self.domain2task = {domain: [task for task in domain2task2slice[domain]] for domain in domain2task2slice}
        self.domain_task = [f'{domain}_{task}' for domain in domain2task2slice for task in domain2task2slice[domain]]
        self.domain_task2idx = {domain_task: idx for idx, domain_task in enumerate(self.domain_task)}
        self.domain_task2idx['all'] = -1
        self.idx2domain_task = {self.domain_task2idx[domain_task]: domain_task for domain_task in self.domain_task2idx.keys()}
        self.idx2domain_task[-1] = 'all'

        self.domain_task2slice = {f'{domain}_{task}': domain2task2slice[domain][task] for domain in domain2task2slice for task in domain2task2slice[domain]}

        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

        self._orig_hw = None
        assert next(iter(domain2task2slice)) in sam_model.domain2task2channel, f"Missing key '{next(iter(domain2task2slice))}'"
    
    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "MDMTSAM2Predictor":
        """
        Load a pretrained model from the Hugging Face hub.

        Arguments:
          model_id (str): The Hugging Face repository ID.
          **kwargs: Additional arguments to pass to the model constructor.

        Returns:
          (SAM2ImagePredictor): The loaded model.
        """
        from sam2.build_sam import build_sam2_hf

        sam_model = build_sam2_hf(model_id, **kwargs)
        return cls(sam_model, **kwargs)
    
    def forward(self, image_list, task_name, if_transform=True):
        # image_list.shape: BHWC
        # task_idx: f'{domain}_{task}' or 'all'
        
        self._orig_hw = image_list.shape[2:]
            
        if if_transform:
            img_batch = self._transforms.forward_batch(image_list).to(self.device)
        else:
            img_batch = image_list
        if len(img_batch.shape) != 4:
            img_batch.unsqueeze(0)
        batch_size = img_batch.shape[0]
        if task_name == 'all':
            batch_size *= len(self.domain_task) 
            task_idx = [-1]
        elif '_' not in task_name:
            batch_size *= len(self.domain2task[task_name])
            task_idx = []
            for k in self.domain_task2idx:
                if task_name in k:
                    task_idx.append(self.domain_task2idx[k])
        else:
            task_idx = [self.domain_task2idx[task_name]]

        backbone_out = self.model.forward_image(img_batch, task_idx)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)

        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        
        # high_res_num x (B*task_num) x C x H x W
        feats = [
            feat.permute(1, 2, 0).reshape(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(None, None, None)

        high_res_feats = feats[:-1]
        
        image_embed = feats[-1]
        if task_idx[0] == -1:
            _, c, h, w = image_embed.shape
            image_embed = image_embed.reshape(len(self.domain_task), -1, c, h, w)
        elif len(task_idx) != 1:
            _, c, h, w = image_embed.shape
            image_embed = image_embed.reshape(len(self.domain2task[task_name]), -1, c, h, w)

        postprocess_masks = {}
        if task_idx[0] == -1:
            for idx in range(len(self.model.sam_mask_decoder)):
                domain_task_embededs = image_embed[idx]
                task_name = self.idx2domain_task[idx]
                postprocess_masks[task_name] = []
                for img_idx in range(len(domain_task_embededs)):
                    domain_task_low_res_masks = self.model.sam_mask_decoder[idx](
                        image_embeddings=domain_task_embededs[img_idx].unsqueeze(0),
                        image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings[self.domain_task2slice[self.idx2domain_task[idx]]],
                        multimask_output=True,
                        repeat_image=False,
                        high_res_features=high_res_feats,
                        img_idx=img_idx,
                    )
                    postprocess_masks[task_name].append(self._transforms.postprocess_masks(domain_task_low_res_masks, self._orig_hw))
                postprocess_masks[task_name] = torch.stack(postprocess_masks[task_name])
        elif len(task_idx) == 1:
            postprocess_masks[task_name] = []
            for img_idx in range(len(image_embed)):
                low_res_masks = self.model.sam_mask_decoder[task_idx[0]](
                    image_embeddings=image_embed[img_idx].unsqueeze(0),
                    image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings[self.domain_task2slice[task_name]],
                    multimask_output=True,
                    repeat_image=False,
                    high_res_features=high_res_feats,
                    img_idx=img_idx,
                )
            
                postprocess_masks[task_name].append(self._transforms.postprocess_masks(
                        low_res_masks, self._orig_hw
                ))
            postprocess_masks[task_name] = torch.stack(postprocess_masks[task_name])
        else:
            for _, task in enumerate(self.domain2task[task_name]):
                domain_task = f'{task_name}_{task}'
                postprocess_masks[domain_task] = []
                idx = self.domain_task2idx[domain_task]
                domain_task_embededs = image_embed[_]
                postprocess_masks[domain_task] = []
                for img_idx in range(len(domain_task_embededs)):
                    domain_task_low_res_masks = self.model.sam_mask_decoder[idx](
                        image_embeddings=domain_task_embededs[img_idx].unsqueeze(0),
                        image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings[self.domain_task2slice[self.idx2domain_task[idx]]],
                        multimask_output=True,
                        repeat_image=False,
                        high_res_features=high_res_feats,
                        img_idx=img_idx,
                    )
                    postprocess_masks[domain_task].append(self._transforms.postprocess_masks(domain_task_low_res_masks, self._orig_hw))
                postprocess_masks[domain_task] = torch.stack(postprocess_masks[domain_task])

        return postprocess_masks

