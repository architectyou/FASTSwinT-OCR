from tqdm import tqdm
import os, cv2

gt_path = '/home/pirl/Desktop/OCR/FAST/data/testset/gts/'
img_path = '/home/pirl/Desktop/OCR/FAST/data/testset/imgs/'
output_dir = '/home/pirl/Desktop/OCR/FAST/data/testset/lpn/'

if not os.path.exists(output_dir+'imgs/'):
    os.makedirs(output_dir+'imgs/')

with open(output_dir+'gts.txt', 'a', encoding = 'utf-8') as txt_file:

    for i in tqdm(os.listdir(img_path)):
        # gt에 img 없으면 pass
        if i.split('.')[0]+'.txt' not in os.listdir(gt_path):
            continue

        with open(gt_path+i.split('.')[0]+'.txt', 'r', encoding ='utf-8') as gt_file:
            lines = gt_file.readlines()

        image = cv2.imread(img_path+i)
        #print(i)
        for line_number, line in enumerate(lines, start=1):
            split_data = line.strip().split(',', maxsplit=9)
            bbox_x = [int(round(float(split_data[i*2]))) for i in range(4)]
            bbox_y = [int(round(float(split_data[i*2+1]))) for i in range(4)]
            #print(bbox_x, bbox_y)

            x_min, x_max = min(bbox_x), max(bbox_x)
            y_min, y_max = min(bbox_y), max(bbox_y)
                
            cropped_img = image[y_min:y_max, x_min:x_max]

            cropped_img_name = f"{i.split('.')[0]}_{line_number}.jpg"
            cropped_img_path = os.path.join(output_dir+'imgs/', cropped_img_name)

                        
            if cropped_img is None or cropped_img.size == 0:
                print(i, line_number,':',line, "is empty !")
                continue

            lang = split_data[-2]
            text = split_data[-1]

            if text == '###' : 
                pass
            else : 
                #print(cropped_img_path)
                cv2.imwrite(cropped_img_path, cropped_img)
                txt_file.write(f"{cropped_img_name}\t{lang}\n")