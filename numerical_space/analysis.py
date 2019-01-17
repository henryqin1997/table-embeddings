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
    dic=dic_asc_fl

    labels=[('16','b', 'o'), ('137','y', '^'),('221','r', 'x')]
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for label,color,marker in labels:
        X = []
        Y = []
        Z = []
        for keys in dic:
            if label in dic[keys]:
                keyArr=list(map(float,keys.split(",")))
                X.append(keyArr[0])
                Y.append(keyArr[1])
                Z.append(keyArr[3]-keyArr[2])
        ax.scatter(X, Y, Z, c=color, marker=marker,label=label)
    ax.set_xlabel('mean')
    ax.set_ylabel('variance')
    ax.set_zlabel('range')
    ax.legend()
    plt.show()
if __name__=="__main__":
    main()
