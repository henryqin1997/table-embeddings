from matplotlib import pyplot as plt
import json

def main():
    filename='data/wordlist_v6.json'
    dic=json.load(open(filename))
    total=0
    onrun=0
    for key in dic.keys():
        total+=dic[key]
    frac=[]
    for key in dic.keys():
        onrun+=dic[key]
        frac.append(onrun/total)
    plt.plot(range(len(frac)),frac)
    plt.show()

def fraction(index):
    filename = 'data/wordlist_v6.json'
    dic = json.load(open(filename))
    total = 0
    onrun = 0
    for key in dic.keys():
        total += dic[key]
    ind=0
    for key in dic.keys():
        ind+=1
        onrun += dic[key]
        if ind>=index:
            break
    print(onrun/total)

if __name__=='__main__':
    fraction(3334)
    #main()