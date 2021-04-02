import json

list=[]
with open('./corpus/news-commentary-v14.en-zh.tsv', 'r', encoding='utf-8') as f:
    for i in f:
        list.append(f.readline().split('\t'))

eng,cn=[],[]
for l in list:
    eng.append(l[0])
    cn.append(l[1])

dict={}
for l in list:
    english = l[0]
    chinese = l[1]
    dict[english]=chinese

with open('./corpus/dict.json', 'w', encoding='utf-8') as f:
    json.dump(dict,f,ensure_ascii=False)



#%% 调取数据
import json
def get_data_from_dict():
    with open('./corpus/dict.json', encoding='utf-8') as f:
        dict =json.load(f)

    eng,ch=[],[]
    for e,c in dict.items():
        eng.append(e)
        ch.append(c)

    return eng,ch

if __name__ == '__main__':
    eng,ch = get_data_from_dict()
    print(len(eng),len(ch))