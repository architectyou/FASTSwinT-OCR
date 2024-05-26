import os, json
import shutil
from tqdm import tqdm
import pdb

# # 이미지파일 경로
# pth_imgs_raw = 'data/1213_random_sample/imgs/'
# pth_imgs = 'data/1213_random_sample/imgs/'

# if not os.path.isdir(pth_imgs):
#     os.mkdir(pth_imgs)

# for i in os.listdir(pth_imgs_raw):
#     if i[-4:]!='json':
#         shutil.copy(os.path.join(pth_imgs_raw, i), pth_imgs)


# json 파일 경로
pth_json = 'data/testset/gts_json/'
pth_txt = 'data/testset/gts/'

for i in os.listdir(pth_json):
    # json 파일 읽기
    if i[-4:]=='json':
        with open(pth_json+i, 'r') as json_file:
            print(json_file)
            data = json.load(json_file)

            txt_data = []
            for j in data['Annotation']:
                # 필요한 값 읽기
                text_language = j['text_language']
                
                bbox_x = []
                bbox_y = []
                
                if len(j['polygon_points']) == 4:
                    for n in range(len(j["polygon_points"])):
                        bbox_x.append(j["polygon_points"][n][0])
                        bbox_y.append(j["polygon_points"][n][1])
                else : 
                    continue
                text = j["text"]

                # language 변환
                lang = ['K','E','M','C','J','MASK']
                c_lang = ['Korean','Latin','Mixed','Chinese','Japanese','Mask']
                text_language = c_lang[lang.index(text_language)]

                # bbox 변환
                bbox = ''
                for c in range(len(j["polygon_points"])):
                    bbox+=str(bbox_x[c])+','+str(bbox_y[c])
                    if c != 3:
                        bbox+=','

                if text_language=='Mixed' or text_language=='Mask' or text=='xxx' or text=='null' or ('×' in text):
                    continue
                else:
                    txt_data.append((bbox, text_language, text))


        # 경로에 gts디렉토리 없으면 만들기
        if not os.path.isdir(pth_txt):
            os.mkdir(pth_txt)

        # txt 파일에 저장
        with open(pth_txt+i[:-4]+'txt', 'w') as txt_file:
            for entry in txt_data:
                line=','.join(map(str, entry))
                txt_file.write(line+"\n")