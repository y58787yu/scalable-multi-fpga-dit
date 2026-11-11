"""Weight/scaling pair supplied to a matrix operation."""

class TP4StageModel:
    def _weight(self,pack,name):return pack[name+'.q'],pack[name+'.s']
