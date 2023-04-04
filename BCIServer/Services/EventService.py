import threading

SessionControlCode = {
    'off': '512',

    'session start': '257',
    'session end':'258',
    'run start':'259',
    'run end':'260',
    'trial start':'261',
    'trial end':'262',

    'record data':'263',
    'request data':'264',
}

ReversedSessionControlCode = dict(zip(SessionControlCode.values(), SessionControlCode.keys()))


class EventService(object):
    _instance_lock = threading.Lock()
    def __init__(self):
        self.cur_sessionControlType = 'off'
        self.last_sessionControlType = self.cur_sessionControlType
        self.typeChangedHandler = {
            'defualt':self.defaultChangedHandler
        }

    def __new__(cls, *args, **kwargs):
        if not hasattr(EventService,'_instance'):
            with EventService._instance_lock:
                if not hasattr(EventService, '_instance'):
                    EventService._instance = object.__new__(cls)
        return EventService._instance

    def defaultChangedHandler(self, curType):
        print('Session Control code changed to: ', curType)

    def MessageHandler(self, msg):
        #print('EventService received message: ', msg)
        msg_split = msg.split('_')
        cmd, code = msg_split[0], msg_split[1]
        if cmd=='SessionControl':
            self.last_sessionControlType = self.cur_sessionControlType
            sessionControlType = ReversedSessionControlCode[code]
            self.cur_sessionControlType = sessionControlType
            if sessionControlType!=self.last_sessionControlType:
                for key in self.typeChangedHandler:
                    self.typeChangedHandler[key](sessionControlType)

