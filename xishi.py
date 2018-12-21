from Common import *
import json

def gamma(image,thre):
    '''
    :param image: numpy type
           thre:float
    :return: image numpy
    '''
    f = image / 255.0
    # we can change thre accoding  to real condition
    # thre = 0.3
    out = np.power(f, thre)

    return out*255.0

def backGamma(image,thre):
    '''
    :param image: numpy type
           thre:float
    :return: image numpy
    '''
    f = image / 255.0
    # thre = 0.2 is best for red
    out = np.power(f, thre)

    return out * 255.0
def HSV(img):

    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV)

    h = H.reshape(H.shape[0]*H.shape[1],order='C')
    s = S.reshape(S.shape[0]*S.shape[1],order='C')
    v = V.reshape(V.shape[0] * V.shape[1], order='C')

    return h,s,v

def GetHsvProperty(h,s,v):

    h_ave = h.mean()
    s_ave = s.mean()
    v_ave = v.mean()
    h_var = h.var()
    s_var = s.var()
    v_var = h.var()

    return h_ave,s_ave,v_ave,h_var,s_var,v_var

#针对红颜色吸湿器
def RedReader(image, info):

    # init below vars ,
    h_ave = -1
    s_ave = -1
    v_ave = -1
    h_var = -1
    s_var = -1
    v_var = -1
    """
    :param image: ROI image
    :param info: information for this meter
    :return: {
       color:
       h_ave:
       s_ave:
       v_ave:
       h_var:
       s_var:
       v_var:
    }
    """
    #Gamma transformation of original image
    image = gamma(image,0.2)

    #the step need
    cv2.imwrite("gamma.jpg",image)
    image = cv2.imread("gamma.jpg")

    meter = meterFinderByTemplate(image, info["template"])

    if meter!=None:
        h,s,v = HSV(meter)
        h_ave,s_ave,v_ave,h_var,s_var,v_var = GetHsvProperty(h,s,v)
        cv2.imwrite("meter.jpg",meter)

    res = {
       'color':-1,
        'h_ave':h_ave,
       's_ave':s_ave,
       'v_ave':v_ave,
       'h_var':h_var,
       's_var':s_var,
       'v_var':v_var
    }
    res = json.dumps(res)

    return res