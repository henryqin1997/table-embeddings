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

if __name__=='__main__':
    main()