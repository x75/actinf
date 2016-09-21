
from naoqi import ALProxy
import time
import numpy as np
import pylab as pl

def plotstuff():
	X__ = np.load("tm_X.npy")
	S_pred = np.load("tm_S_pred.npy")
	E_pred = np.load("tm_E_pred.npy")
	M = np.load("tm_M.npy")

	pl.ioff()
	pl.suptitle("mode: %s (X: FM input, state pred: FM output)" % ("bluib"))
	pl.subplot(511)
	pl.title("X[goals]")
	pl.plot(X__[10:,0:4], "-x")
	pl.subplot(512)
	pl.title("X[prediction error]")
	pl.plot(X__[10:,4:], "-x")
	pl.subplot(513)
	pl.title("state pred")
	pl.plot(S_pred)
	pl.subplot(514)
	pl.title("error state - goal")
	pl.plot(E_pred)
	pl.subplot(515)
	pl.title("state")
	pl.plot(M)
	pl.show()

IP = "127.0.0.1"
PORT = 9559
    
motionProxy = ALProxy("ALMotion", IP, PORT)
    
joint_names = ["LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll"]
values = []
    
numsteps = 1000


bodyNames = motionProxy.getBodyNames("Body")
print "Body:"
print str(bodyNames)
print ""

from sklearn.neighbors import KNeighborsRegressor

X_ = []
y_ = []
mdl = KNeighborsRegressor(n_neighbors=5)
for i in range(10):
    X_.append(np.random.uniform(-0.1, 0.1, (8,)))
    y_.append(np.random.uniform(-0.1, 0.1, (4,)))
        # print X_, y_
mdl.fit(X_, y_)

goal = np.random.uniform([-2.0857, -0.3142, -2.0857, -1.5446], [2.0857, 1.3265, 2.0857, -0.0349], (1, 4)) * 0.4
e_pred = np.zeros((1, 4))

S_pred = np.zeros((numsteps, 4))
E_pred = np.zeros((numsteps, 4))
M      = np.zeros((numsteps, 4))


motionProxy.setStiffnesses(joint_names, 1.0)


for i in range(numsteps):
    X = np.hstack((goal, e_pred)) # model input: goal and prediction error
    s_pred = mdl.predict(X) # state prediction

    joint_angles = np.clip(s_pred, [-2.0857, -0.3142, -2.0857, -1.5446], [2.0857, 1.3265, 2.0857, -0.0349]).flatten().tolist()

    motionProxy.setAngles(joint_names, joint_angles, 0.4)
    time.sleep(0.002)
    m_ = motionProxy.getAngles(joint_names, True)
    m =np.array(m_)

    e_pred = m - goal
    tgt = s_pred - (e_pred * 0.5)

    X_.append(X[0,:])
    y_.append(tgt[0,:])

    mdl.fit(X_, y_)
            
    # print s_pred
    # print "X.shape, s_pred.shape, e_pred.shape, m.shape, s_ext.shape", X.shape, s_pred.shape, e_pred.shape, m.shape, s_ext.shape

    S_pred[i] = s_pred
    E_pred[i] = e_pred
    M[i]      = m
    
    if i % 50 == 0:
        # goal = np.random.uniform(-np.pi/2, np.pi/2, (1, environment.conf.m_ndims))
	goal = np.random.uniform([-2.0857, -0.3142, -2.0857, -1.5446], [2.0857, 1.3265, 2.0857, -0.0349], (1, 4)) * 0.4
        # goal = np.random.uniform(-1.0, 1.0, (1, 4))
        print "new goal[%d] = %s" % (i, goal)
        print "e_pred = %f" % (np.linalg.norm(e_pred, 2))

    # print type(m)
    # new_angles = np.random.uniform(-0.1, 0.1, (4,)).tolist()
    time.sleep(0.098)

    X__ = np.array(X_)
    np.save("tm_X.npy", X__)
    np.save("tm_S_pred.npy", S_pred)
    np.save("tm_E_pred.npy", E_pred)
    np.save("tm_M.npy", M)


plotstuff()

