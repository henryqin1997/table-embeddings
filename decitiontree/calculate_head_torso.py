import json

def main():
    for sample in range(10):
        head_correct=0
        head_num=0
        torso_correct=0
        torso_num=0
        diction = json.load(open("sample_dict_it={}".format(sample)))
        for key in diction.keys():
            if int(key)<=200:
                head_correct+=diction[key][0]
                head_correct += diction[key][2]
                head_num+=diction[key][1]
                head_num+=diction[key][3]
            elif key!='4510':
                torso_correct+=diction[key][0]
                torso_correct+=diction[key][2]
                torso_num+=diction[key][1]
                torso_num+=diction[key][3]

        print("sample {}".format(sample))
        print(head_correct)
        print(head_num)
        print("head accuracy: {}".format(head_correct/head_num))
        print(torso_correct)
        print(torso_num)
        print("torso accuracy: {}".format(torso_correct/torso_num))

    head_correct = 0
    head_num = 0
    torso_correct = 0
    torso_num = 0
    correct = 0
    num = 0
    dic=json.load(open('sample_table_pred'))
    for key in dic.keys():
        if key!='4510':
            correct+=dic[key][0]+dic[key][2]
            num+=dic[key][1]+dic[key][3]
        if int(key) <= 200:
            head_correct += dic[key][0]
            head_correct += dic[key][2]
            head_num += dic[key][1]
            head_num += dic[key][3]
        elif key != '4510':
            torso_correct += dic[key][0]
            torso_correct += dic[key][2]
            torso_num += dic[key][1]
            torso_num += dic[key][3]

    print("table")
    print('accuracy {}'.format(float(correct/num)))
    print(head_correct)
    print(head_num)
    print("head accuracy: {}".format(head_correct / head_num))
    print(torso_correct)
    print(torso_num)
    print("torso accuracy: {}".format(torso_correct / torso_num))



if __name__=='__main__':
    main()
