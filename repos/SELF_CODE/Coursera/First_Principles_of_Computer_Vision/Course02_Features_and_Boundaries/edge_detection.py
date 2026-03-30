import  numpy           as      np


##########################################################################
# 1D Edge Detection
##########################################################################
def edge_detection_1d(signal, autothresold = True, threshold = 1):  
    '''
        Edge Detection for 1D signal
        Input:
            . signal        : 1D signal
            . autothreshold : auto calculate thresold
            . threshold     : if autothreshold = False, use pre-defined threshold
        Reuturn
            . Indices of egde
    '''
    edge_indices        = []
    

    #---------------------------------------------------------------------
    # Statistic the change if autothreshold
    #---------------------------------------------------------------------
    if autothresold:
        #-----------------------------------------------------------------
        # Changes in signal
        changes     =   signal[1:] - signal[:-1]
        changes     =   [abs(x) for x in changes]

        #-----------------------------------------------------------------
        # Outlier as threshold



    #---------------------------------------------------------------------
    # Iterate over the signal
    #---------------------------------------------------------------------
    for i in range(1, len(signal) - 1):

        #-----------------------------------------------------------------
        # Check for an edge by comparing adjacent values
        if signal[i] - signal[i-1] >= threshold and signal[i] - signal[i+1] >= threshold:
            edge_indices.append(i)
    
    return edge_indices


##########################################################################
# Testing part
##########################################################################
def main():
    #---------------------------------------------------------------------
    # 1D Edge detection
    #---------------------------------------------------------------------
    signal      =   [1, 2, 3, 2, 1, 4, 3, 2, 1]
    edges       =   edge_detection_1d(signal)

    print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
    print('1D Signal Detection')
    print('Input signal: ', signal)  
    print("Edge indices:", edges)
    print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')


if __name__ == "__main__":
    main()