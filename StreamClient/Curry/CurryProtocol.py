import numpy as np
from StreamClient.Curry.utils import *

CurryChannelTypes = {
    0: 'eeg',
    100: 'stim'
}

def controlCode(type):
    i = -1
    if type=='CTRL_FromServer':
        i = 1
    elif type=='CTRL_FromClient':
        i = 2
    return i

def requestType(type):
    r = -1
    if type=='RequestVersion':
        r = 1
    elif type=='RequestChannelInfo':
        r = 3
    elif type=='RequestBasicInfoAcq':
        r = 6
    elif type=='RequestStreamingStart':
        r = 8
    elif type=='RequestStreamingStop':
        r = 9
    elif type=='RequestAmpConnect':
        r = 10
    elif type=='RequestAmpDisconnect':
        r = 11
    elif type=='RequstDelay':
        r = 16
    return r

def dataType(type):
    d = -1
    if type=='Data_Info':
        d = 1
    elif type=='Data_Eeg':
        d = 2
    elif type=='Data_Events':
        d = 3
    elif type=='Data_Impedance':
        d = 4
    return d

def infoType(type):
    i = -1
    if type=='InfoType_Version':
        i = 1
    elif type=='InfoType_BasicInfo':
        i = 2
    elif type=='InfoType_ChannelInfo':
        i = 4
    elif type=='InfoType_StatusAmp':
        i = 7
    elif type=='InfoType_Time':
        i = 9
    return i

def blockType(type):
    d = -1
    if type=='DataTypeFloat32bit':
        d = 1
    elif type=='DataTypeFloat32bitZIP':
        d = 2
    elif type=='DataTypeEventList':
        d = 3
    return d

def initHeader(chanID, code, request, samples, sizeBody, sizeUn):
    c_chID = [ord(c) for c in chanID]
    w_Code = [c for c in np.array(code,dtype=np.uint16).byteswap(inplace=True).tobytes()]
    w_Requst = [c for c in np.array(request,dtype=np.uint16).byteswap(inplace=True).tobytes()]
    un_Sample = [c for c in np.array(samples,dtype=np.uint32).byteswap(inplace=True).tobytes()]
    un_Size = [c for c in np.array(sizeBody,dtype=np.uint32).byteswap(inplace=True).tobytes()]
    un_SizeUn = [c for c in np.array(sizeUn,dtype=np.uint32).byteswap(inplace=True).tobytes()]

    header = c_chID;header.extend(w_Code);header.extend(w_Requst);header.extend(un_Sample);header.extend(un_Size);header.extend(un_SizeUn)
    return header

def requestPacket(con, method, packetSize, handles):
    con.settimeout(1)
    try:
        data = con.recv(packetSize)
        return (data, 1)
    except:
        return (None, 0)

def clientProcessRequest(con, handles, header, method, code, request, init):
    headerSize = len(header)
    if headerSize!=20:
        print('Error in Header length')

    if init is False:
        con.send("".join([chr(s) for s in header]).encode())

    #Receive header
    [data, status] = requestPacket(con=con, method=method, packetSize=20, handles=handles)

    tempPacketSize = 0
    count = 0
    timeout = 10
    synchPackets = 5
    dataOut = []
    message = {'code':None,
               'request':None,
               'startSample':None,
               'packetSize':None}
    if status:
        message['code'] = int.from_bytes(np.flip(data[4:6]).tobytes(),'big')
        message['request'] = int.from_bytes(np.flip(data[6:8]).tobytes(),'big')
        message['startSample'] = int.from_bytes(np.flip(data[8:12]).tobytes(),'big')
        message['packetSize'] = int.from_bytes(np.flip(data[12:16]).tobytes(),'big')
        # Receive body
        if type(code) is not list:
            code = [code]

        if type(request) is not list:
            request = [request]

        if(message['code'] in code) and (message['request'] in request):
            # in case packet comes in separate chunks
            while (tempPacketSize < message['packetSize']) and (count < timeout):
                [data, status] = requestPacket(con, method, message['packetSize'], handles)
                tempPacketSize = tempPacketSize + len(data)
                dataOut.append(data)
                count = count + 1
            dataOut = b''.join(dataOut)
            return [dataOut, status, message]
        else:
            if(message['request'] not in request) and method=='ClientRequestDataPacket':
                print(' WARNING: Verify data format! This demo only supports uncompressed data.')
            print(method,' failed: code or request')
            print('Attempting synchronization...')
            count = 0

            while(count<synchPackets):
                requestPacket(con,['SynchPacket (',str(count+1),' of ', str(synchPackets),')'],message['packetSize'],handles)
                count = count + 1

            return [data, 0, message]
    else:
        return [data, 0, message]

def clientCheckNetStreamingVersion(con):
    NetStreamingVersion = 802

    header = initHeader('CTRL',controlCode('CTRL_FromClient'),requestType('RequestVersion'),0,0,0)
    [version_bytes, status, message] = clientProcessRequest(con=con, handles=None, header=header, method= 'NetStreamingVersion',
                                             code=dataType('Data_Info'),request=infoType('InfoType_Version'),init=False)
    version = int.from_bytes(version_bytes,'little')
    if status:
        if version==NetStreamingVersion:
            print('NetStreaming version',version)
            print('CheckNetStreamingVersion successful')
        else:
            print('CheckNetStreamingVersion failed: incompatible client version')
    return [status, version]

def clientGetBasicInfo(con, handles):
    basicInfo = []
    maxChans = 300

    header = initHeader(chanID='CTRL',
                        code=controlCode('CTRL_FromClient'),
                        request=requestType('RequestBasicInfoAcq'),
                        samples=0,sizeBody=0,sizeUn=0)
    [basicInfoRaw, status, message] = clientProcessRequest(con=con,
                                                  handles=handles,
                                                  header=header,
                                                  method='ClientGetBasicInfo',
                                                  code=dataType('Data_Info',),
                                                  request=infoType('InfoType_BasicInfo'),
                                                  init=False)
    if status:
        basicInfo = {
            'size': int.from_bytes(basicInfoRaw[0:4],byteorder='little'),
            'eegChan':int.from_bytes(basicInfoRaw[4:8],byteorder='little'),
            'sampleRate':int.from_bytes(basicInfoRaw[8:12],byteorder='little'),
            'dataSize':int.from_bytes(basicInfoRaw[12:16],byteorder='little'),
            'allowClientToControlAmp':int.from_bytes(basicInfoRaw[16:20],byteorder='little'),
            'allowClientToControlRec':int.from_bytes(basicInfoRaw[20:24],byteorder='little'),
        }

    if (basicInfo['eegChan']>0) and (basicInfo['eegChan']<maxChans) and (basicInfo['sampleRate']>0) and (basicInfo['dataSize']==2 or basicInfo['dataSize']==4):
        print(basicInfo['eegChan'],' Channels')
        print(basicInfo['sampleRate'], ' Hz Sample Rate')
        if basicInfo['allowClientToControlAmp']:
            print(' Client is allowed to control amplifier')
        else:
            print(' Client is not allowed to control amplifier')
        print('ClientGetBasicInfo successful')
        return [status, basicInfo]
    else:
        print('Error in ClientGetBasicInfo data')
        return [status, basicInfo]

def clientGetChannelInfoList(con, numChannels, handles):
    # Offsets in CURRY struct (in bytes)

    offset_channelId = 0
    offset_chanLabel = offset_channelId + 4
    offset_chanType = offset_chanLabel + 80
    offset_deviceType = offset_chanType + 4
    offset_eegGroup = offset_deviceType + 4
    offset_posX = offset_eegGroup + 4
    offset_posY = offset_posX + 8
    offset_posZ = offset_posY + 8
    offset_posStatus = offset_posZ + 8
    offset_bipolarRef = offset_posStatus + 4
    offset_addScale = offset_bipolarRef + 4
    offset_isDropDown = offset_addScale + 4
    offset_isNoFilter = offset_isDropDown + 4

    chanInfoLen = (offset_isNoFilter + 4) - 1 # Raw length
    chanInfoLen = round((chanInfoLen+1)/8) * 8  # Length of CURRY channel info struct in bytes, consider padding

    infoList = []

    header = initHeader(chanID='CTRL',
                        code=controlCode('CTRL_FromClient'),
                        request=requestType('RequestChannelInfo'),
                        samples=0,
                        sizeBody=0,
                        sizeUn=0)

    [infoListRaw, status, message] = clientProcessRequest(con=con, handles=handles,
                                                 header=header, method='ClientGetChannelInfoList',
                                                 code=dataType('Data_Info'),request=infoType('InfoType_ChannelInfo'),
                                                 init=False)

    if status:
        for i in range(1,numChannels+1):
            j = chanInfoLen * (i-1)
            infoList.append([
                {
                    'id':int.from_bytes(infoListRaw[j+offset_channelId: j+(offset_chanLabel)],byteorder='little'),
                    'chanLabel': typecast_uint16_from_bytes(infoListRaw[j + offset_chanLabel: j + (offset_chanType)]),
                    'chanType': int.from_bytes(infoListRaw[j + offset_chanType: j + (offset_deviceType)],byteorder='little'),
                    'deviceType': int.from_bytes(infoListRaw[j + offset_deviceType: j + (offset_eegGroup)],byteorder='little'),
                    'eegGroup': int.from_bytes(infoListRaw[j + offset_eegGroup: j + (offset_posX)],byteorder='little'),
                    'posX': typecast_double_from_bytes(infoListRaw[j + offset_posX: j + (offset_posY)])[0],
                    'posY': typecast_double_from_bytes(infoListRaw[j + offset_posY: j + (offset_posZ)])[0],
                    'posZ': typecast_double_from_bytes(infoListRaw[j + offset_posZ: j + (offset_posStatus)])[0],
                    'posStatus': int.from_bytes(infoListRaw[j + offset_posStatus: j + (offset_bipolarRef)],byteorder='little'),
                    'bipolarRef': int.from_bytes(infoListRaw[j + offset_bipolarRef: j + (offset_addScale)],byteorder='little'),
                    'addScale': typecast_float_from_bytes(infoListRaw[j + offset_addScale: j + (offset_isDropDown)])[0],
                    'isDropDown': int.from_bytes(infoListRaw[j + offset_isDropDown: j + (offset_isNoFilter)],byteorder='little'),
                    'isNoFilter': typecast_uint32_from_bytes(infoListRaw[j + offset_isNoFilter:chanInfoLen*i]),
                }
            ])

        for i in range(1,numChannels+1):
            chan_name = infoList[i-1][0]['chanLabel']
            infoList[i-1][0]['chanLabel'] = ''.join([chr(c) for c in [c for c in chan_name if c!=0]])

        print('Labels: ', [info[0]['chanLabel'] for info in infoList])
        print('ClientGetChannelInfoList successful')

        return [status, infoList]

    else:
        return [0, infoList]

def clientRequestDataPacket(con, basicInfo, infoList, handles, startStop, init, verbose=False):
    """
    Can receive eeg data, events and impedance values.

    Event variables definitions
    C++ definition of CURRY event struct:
    struct NetStreamingEvent
      {                                 % Offsets in bytes
        long	nEventType;             % 1:4
        long	nEventLatency;          % 5:8
        long	nEventStart;            % 9:12
        long	nEventEnd;              % 13:16
        wchar_t	wcEventAnnotation[260]; % 17:536
      };

    :param con:
    :param basicInfo:
    :param infoList:
    :param handles:
    :param startStop: 1 if start, 0 if stop
    :param init:
    :return:
    """

    #Offsets for struct variables
    offsetEventType = 0
    offsetEventLatency = offsetEventType + 4
    offsetEventStart = offsetEventLatency + 4
    offsetEventEnd = offsetEventStart + 4
    offsetEventAnnotation = offsetEventEnd + 4

    eventStructLength = (offsetEventAnnotation+1 + 520) - 1 #Raw length
    eventStructLength = round(eventStructLength/8)*8        #Length of CURRY event struct in bytes, consider padding

    dataTypes = [dataType('Data_Eeg'), dataType('Data_Events'), dataType('Data_Impedance')]

    blockTypes = [blockType('DataTypeFloat32bit'), blockType('DataTypeEventList')]

    if startStop:
        header = initHeader(chanID='CTRL',
                            code=controlCode('CTRL_FromClient'),
                            request=requestType('RequestStreamingStart'),
                            samples=0,
                            sizeBody=0,
                            sizeUn=0)
    else:
        header = initHeader(chanID='CTRL',
                            code=controlCode('CTRL_FromClient'),
                            request=requestType('RequestStreamingStop'),
                            samples=0,
                            sizeBody=0,
                            sizeUn=0)

    [data, status, message] = clientProcessRequest(con=con, handles=handles, header=header,
                                                   method='ClientRequestDataPacket',
                                                   code=dataTypes, request=blockTypes, init=init)

    dataList = {}

    if startStop is False:
       return [status, data]

    if status:
        if message['code']==2:   #eeg data
            receivedSamples = len(data) / (basicInfo['dataSize']*basicInfo['eegChan'])
            if verbose:
                print('Received '+str(len(data)/1000)+' kBytes EEG, '+str(receivedSamples)+' samples, Start sample= '+str(message['startSample']))
            packet = getPacket(handles=handles,basicInfo=basicInfo,infoList=infoList,data=data)
            dataList['eeg'] = packet
            dataList['receivedSamples'] = receivedSamples
            dataList['startSample'] = message['startSample']
        elif message['code']==3:    #event data
            if message['packetSize']%eventStructLength==0:
                numEvents = int(message['packetSize']/eventStructLength)
                if verbose:
                    print('Received '+str(len(data))+' Bytes, Event ')
                if numEvents>0:
                    for i in range(0,numEvents):
                        eventOffset = eventStructLength * i
                        eventType = int.from_bytes(data[eventOffset+offsetEventType:eventOffset+offsetEventLatency-1],'little')
                        eventLatency = int.from_bytes(data[eventOffset+offsetEventLatency:eventOffset+offsetEventStart -1],byteorder='little')
                        #eventAnnotation = int.from_bytes(data[eventOffset+offsetEventAnnotation:eventOffset+eventStructLength],byteorder='little')
                        if verbose:
                            print('Event type:'+str(eventType)+' Lantency:'+str(eventLatency))
                        dataList['event'] = eventType
            else:
                print('ClientRequestDataPacket failed: unmatching event structure size')
                return [0, dataList]
        elif message['code']==4:
            if verbose:
                print('Received '+str(len(data))+' Bytes, Impedance')
    if verbose:
        print('ClientRequestDataPacket successful')
    return [status, dataList]

def getPacket(handles, basicInfo, infoList, data):
    numSamples = int(len(data) / (basicInfo['dataSize']*basicInfo['eegChan']))
    if basicInfo['dataSize']==2:    #int
        data = np.array(typecast_int16_from_bytes(data))
        packet = np.reshape(data,[numSamples, basicInfo['eegChan']]).swapaxes(0,1)
    elif basicInfo['dataSize']==4:  #short
        data = np.array(typecast_single_from_bytes(data))
        packet = np.reshape(data,[numSamples, basicInfo['eegChan']]).swapaxes(0,1)
    else:
        packet = None
        print('Error in plotPacket: Unknown datatype')
    return packet