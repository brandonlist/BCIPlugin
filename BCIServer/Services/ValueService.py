ValueDefine = {
    'MIstate':{'type':int, 'keyValue':{1:'',2:'',3:'',4:'', 5:''}},
    'SSVEPstate':{'type':int, 'keyValue':{1:'',2:'',3:'',4:''}},
    'P300state': {'type': int, 'keyValue': {1: '', 2: '', 3: '', 4: ''}},
}


class ValueService():
    """

    - Should not be designed as Singleton, since multiple stream client is permitted
    """
    def __init__(self):
        self.values = {}
        for name in ValueDefine:
            self.values[name] = None

    def MessageHandler(self, msg):
        msg_split = msg.split('_')
        if msg_split[0]=='Value':
            print('ValueService received message: ', msg)
            name, code = msg_split[1], msg_split[2]
            if name in ValueDefine.keys():
                self.SetValue(name=name, value=code)

    def SetValue(self, name, value):
        self.values[name] = ValueDefine[name]['type'](value)
        print('value ', name,' updated to:', value)

    def UpdateValue(self, name, value, conn):
        self.SetValue(name=name, value=value)
        msg = 'Value_' + str(name) + '_' + str(value)
        conn.sendall(str.encode(msg))
        print('Sending value update:',name,'->',value)


