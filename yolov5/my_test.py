import os

img_file = '/home/zichen/Documents/Project/YOLO/datasets/coco128/images/train2017/000000000009.jpg'
sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'
print(img_file.rsplit(sa, 1))
print(sb.join(img_file.rsplit(sa, 1)))
a = sb.join(img_file.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt'

print(a)