import json

def main():
    for sample in range(10):
        head_correct=0
        head_num=0
        torso_correct=0
        torso_num=0
        diction = json.load(open("sample_dict_it={}".format(sample)))
        for key in diction.keys():
            if int(key)<=1000:
                head_correct+=diction[key][0]
                head_num+=diction[key][1]
            elif key!='4510':
                torso_correct+=diction[key][0]
                torso_num+=diction[key][1]
        print("sample {}".format(sample))
        print(head_correct)
        print(head_num)
        print("head accuracy: {}".format(head_correct/head_num))
        print(torso_correct)
        print(torso_num)
        print("torso accuracy: {}".format(torso_correct/torso_num))

if __name__=='__main__':
    main()
