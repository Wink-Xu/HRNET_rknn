import cv2
import os


if __name__ == "__main__":
    quant_img = r'../../output/quant_img_orin'
    output_dir = r'../../output/quant_img'
    dataset_path = r'./dataset.txt'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir) 
    print(quant_img)
    with open(dataset_path, 'w') as fw:
        for i in os.listdir(quant_img):
            if i.endswith('.jpg'):
                img_name = os.path.join(quant_img, i)
                print(img_name)
                img = cv2.imread(img_name)
                dst_img = cv2.resize(img, (256,192))
                dst_img_name = os.path.join(output_dir, i)
                cv2.imwrite(dst_img_name, dst_img)
            fw.write(dst_img_name)
            fw.write('\n')
    
