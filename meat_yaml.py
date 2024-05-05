#%%
import yaml

data = {
    'train': 'E:/Final_Phase/MeatClassification/dataset/train', 
    'val': 'E:/Final_Phase/MeatClassification/dataset/valid',  
    'test': 'E:/Final_Phase/MeatClassification/dataset/test',  

    # 클래스 개수
    'nc': 1,    
    'names': ['meat']  
}

with open('E:/Final_Phase/MeatClassification/dataset/meat.yaml', 'w') as file:
    yaml.dump(data, file, sort_keys=False)

# %%
