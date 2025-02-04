import os
import json

from torch.utils.data import Dataset
from PIL import Image

from utils import pre_caption

def finetune_dataset_format(metadata_path:str, generated_cap_path:str, output_path:str=None):
    """
    A consistent dict format is required for finetuning, the following will be done:

    Merge true captions into generated captions under "True" field, and add file_path.
    The new dict will be written to output_path if provided, otherwise dict will be returned.

    Args:
        metadata_path: Path to metadata json file (artpedia)
        generated_cap_path: Path to generated captions file (obtained from generate_captions.py)
        output_path: Location to write new dict to
    """

    if output_path!=None and os.path.exists(output_path):
        raise FileExistsError

    with open(metadata_path, "r", encoding='utf-8') as f:
        metadata = json.load(f)
    with open(generated_cap_path, "r", encoding='utf-8') as f2:
        generated = json.load(f2)
    for id, sample in generated.items():
        sample["True"] = " ".join(metadata[id]["visual_sentences"])
        sample["file_path"] = metadata[id]["file_path"]

    if output_path:
        with open(output_path, "w", encoding='utf-8') as f3:
            f3.write(json.dumps(generated, indent=4, ensure_ascii=False))
    else:
        return generated
    
class artpedia_train(Dataset):
    def __init__(self, transform, ann_path, captions_path, captioner, max_words=30, prompt=''):        
        '''
        ann_path (string): Path to the annotation file
        captions_path (string): Path to generated captions file
        captioner (string): Which captioner to use (e.g., 'LlamaCaptioner', 'LlavaCaptioner', 'True')
        '''
        # Merge annotations and captions
        self.merged_data = finetune_dataset_format(ann_path, captions_path)
        self.transform = transform
        self.max_words = max_words
        self.prompt = prompt
        self.captioner = captioner
        
        # Create sequential IDs for all entries
        self.img_ids = {img_id: idx for idx, img_id in enumerate(self.merged_data.keys())}
        
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, index):
        img_id = list(self.img_ids.keys())[index]
        ann = self.merged_data[img_id]
        
        image = Image.open(ann['file_path']).convert('RGB')   
        image = self.transform(image)
        
        caption = ann.get(self.captioner, '')
        caption = self.prompt + pre_caption(caption, self.max_words)

        return image, caption, self.img_ids[img_id]


class artpedia_eval(Dataset):
    def __init__(self, transform, ann_path, captions_path, captioner):
        '''
        ann_path (string): Path to annotation file
        captions_path (string): Path to generated captions file
        captioner (string): Which captioner to use (e.g., 'LlamaCaptioner', 'LlavaCaptioner', 'True')
        '''
        # Merge annotations and captions
        self.merged_data = finetune_dataset_format(ann_path, captions_path)
        self.transform = transform
        self.captioner = captioner
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        img_id = 0
        
        for key, ann in self.merged_data.items():
            self.image.append(ann['file_path'])
            self.img2txt[img_id] = []
            
            caption = ann.get(self.captioner, '')
            self.text.append(pre_caption(caption))
            self.img2txt[img_id].append(txt_id)
            self.txt2img[txt_id] = img_id
            txt_id += 1
            img_id += 1
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    
        image = Image.open(self.image[index]).convert('RGB')    
        image = self.transform(image)  

        return image, index
    
    
if __name__=="__main__":
    train_dataset = artpedia_train(
        transform=None,
        ann_path='artpedia/artpedia_train.json',
        captions_path='captions/train_captions.json',
        captioner='LlavaCaptioner'
    )

    eval_dataset = artpedia_eval(
        transform=None,
        ann_path='artpedia/artpedia_val.json',
        captions_path='captions/val_captions.json',
        captioner='LlavaCaptioner'
    )