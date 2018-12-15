import numpy as np
import time
def getAction(probs,acts):
    acts = np.asarray(acts)
    probs = np.asarray(probs)
    move = np.random.choice(acts,p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
    return move

while(1):
    time.sleep(9)
    probs = []
    acts = []
    move = []
    f = open('question.txt', 'r')  # 源csv文本
    try:
        line = f.readline()
        signal = line
        print(signal)
        line = f.readline()
        array = line.split('|')
        for i in array:
            ispl=i.split(',')
            probs.append(float(ispl[0]))
            acts.append(int(ispl[1])*15+int(ispl[2]))
            move = getAction(probs,acts)
    except BaseException as err:
        print('question File error:',str(err))
    finally:
        f.close()
    output = open('answer.txt', 'w')
    try:
        if(signal.__sizeof__() != 0):
            print(signal+":ans writing..."+str(move))
            output.write(signal)
            output.write(str(move))
            output.close()
    except BaseException as err:
        print('answer File error',str(err))
    finally:
        output.close()
