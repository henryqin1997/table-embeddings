import json

def main():
    for sample in range(10):
        harbo_correct=0
        harbo_num=0
        torso_correct=0
        torso_num=0
        diction = json.load(open("sample_dict_it={}".format(sample)))
        for key in diction.keys():
            if int(key)<=1000:
                harbo_correct+=diction[key][0]
                harbo_num+=diction[key][1]
            elif key!='4510':
                torso_correct+=diction[key][0]
                torso_num+=diction[key][1]
        print("sample {}".format(sample))
        print(harbo_correct)
        print(harbo_num)
        print("harbo accuracy: {}".format(harbo_correct/harbo_num))
        print(torso_correct)
        print(torso_num)
        print("torso accuracy: {}".format(torso_correct/torso_num))

if __name__=='__main__':
    main()
