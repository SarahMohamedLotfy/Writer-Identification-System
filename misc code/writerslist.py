
import os
from shutil import copy2


path = 'D:\\pattern dataset\\xml'
test_cases_path = 'D:\\pattern dataset\\IAMdataset'
images_path = 'D:\\pattern dataset\\forms'
padding = 3
dir_iterator = 1
string_to_search = "<form created"
writer_ids = []
for filename in os.listdir(path):
     with open(os.path.join(path, filename), 'r') as read_obj:
         for line in read_obj:
             if string_to_search in line:
                 leftline,rightline = line.split('writer-id="')
                 answer,junk = rightline.split('">')
                 if (answer != '000'):
                     filename,extension = filename.split('.')
                     item = [answer,filename]
                     writer_ids.append(item)
                 break
                 
                 
writer_ids.sort()
# for item in writer_ids:
#     if item[0] == '000':
#         writer_ids.remove(item)

filename_map = {}
for idnum,name in writer_ids:
    if idnum in filename_map.keys():
        filename_map[idnum].append(name)
    else:
        filename_map[idnum] = [name]

newDict = dict(filter(lambda elem: len(elem[1]) >= 3,filename_map.items()))

keyslist = list(newDict.keys())
final_itr = len(keyslist)//3 
for i in range(0,len(keyslist),3):
    if (i+1 == final_itr):
        break
    directory = format(dir_iterator, '03d')
    dir_iterator+=1
    full_path = os.path.join(test_cases_path,directory)
    os.mkdir(full_path)

    for j in range (1,4):
        newpath = os.path.join(full_path,str(j))
        os.mkdir(os.path.join(full_path,str(j)))
        image_name1 = newDict[keyslist[i+j-1]][0] + '.png'
        image_name2 = newDict[keyslist[i+j-1]][1] + '.png'
        copy2(os.path.join(images_path,image_name1), os.path.join(newpath,'1.png'))
        copy2(os.path.join(images_path,image_name2), os.path.join(newpath,'2.png'))
    test_image = newDict[keyslist[i]][2] + '.png'
    copy2(os.path.join(images_path,test_image),os.path.join(full_path,'test.png'))

# file_writer = open(path+"\\writers.txt","w")
# for item in writer_ids:
#         file_writer.write(item[0] + " " + item[1] + "\n")
    
    