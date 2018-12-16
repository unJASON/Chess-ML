import numpy as np
import time
import os
def getAction(probs,acts):
    acts = np.asarray(acts)
    probs = np.asarray(probs)
    move = np.random.choice(acts,p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
    return move

while(1):
    time.sleep(9)
    probs = []
    acts = []
    if not os.path.exists("chess/question.txt"):
        print('waiting')
        continue
    f = open('chess/question.txt', 'r')  # 源csv文本
    try:
        line = f.readline()
        signal = line[:-1]
        signalspl = signal.split(',')
        print(signal)
        line = f.readline()
        array = line.split('|')
        for i in array:
            ispl=i.split(',')
            probs.append(float(ispl[2])/float(signalspl[2]))
            acts.append(int(ispl[0])*15+int(ispl[1]))
        move = getAction(probs,acts)
    except BaseException as err:
        print('question File error:',str(err))
    finally:
        f.close()
    output = open('chess/'+str(signal)+'.txt', 'w')
    try:
        if(signal.__sizeof__() != 0):
            print(signal+":ans writing..."+str(move))
            output.write(str(move))
            output.close()
    except BaseException as err:
        print('answer File error',str(err))
    finally:
        output.close()