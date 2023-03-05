
from arm.Arm import Arm
class Ads_data(Arm):
    def __init__(self,data,number):
        self.data=data.values[:,number]
        self.length=len(data)
        self.number=number
        self.expectation=sum(self.data)/self.length/10

    # def shape(self):
    #     return self.data.shape[1]

    def draw(self,n):

        reward=self.data[n]
        reward=reward/10

        return reward
