'''
Analysis the data points in the six dictions.
'''
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def main():
    with open('numerical_space/numerical_dic_1_float.json','r') as fp:
        dic_asc_fl = json.load(fp)
    with open('numerical_space/numerical_dic_0_float.json','r') as fp:
        dic_ran_fl = json.load(fp)
    with open('numerical_space/numerical_dic_-1_float.json','r') as fp:
        dic_des_fl = json.load(fp)
    with open('numerical_space/numerical_dic_1_int.json','r') as fp:
        dic_asc_int = json.load(fp)
    with open('numerical_space/numerical_dic_0_int.json','r') as fp:
        dic_ran_int = json.load(fp)
    with open('numerical_space/numerical_dic_-1_int.json','r') as fp:
        dic_des_int = json.load(fp)
    dics=[dic_asc_fl,dic_ran_fl,dic_des_fl,dic_asc_int,dic_ran_int,dic_des_int]
    titles=["dic_1_float","dic_0_float","dic_-1_float","dic_1_int","dic_0_int","dic_-1_int"]
    i=0# index of dic

    labels=[('26','b', 'o'), ('1023','y', '^'),('2357','r', 'x')]
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for label,color,marker in labels:
        X = []
        Y = []
        Z = []
        for keys in dics[i]:
            if label in dics[i][keys]:
                keyArr=list(map(float,keys.split(",")))
                X.append(keyArr[0])
                Y.append(keyArr[1])
                Z.append(keyArr[3]-keyArr[2])
        ax.scatter(X, Y, Z, c=color, marker=marker,label=label)
    ax.set_xlabel('mean')
    ax.set_ylabel('variance')
    ax.set_zlabel('range')
    plt.title(titles[i])
    ax.legend()
    plt.show()
if __name__=="__main__":
    main()
