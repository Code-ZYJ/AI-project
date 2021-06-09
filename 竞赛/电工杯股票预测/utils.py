def get_rate(rate):
        rate = rate*100
        if rate<15:
                if rate%1>0:
                        rate = int(rate)+1
                else:
                        rate = int(rate)
        elif 15<rate<=20:
                rate = 20
        elif 20<rate<=30:
                rate = 30
        elif 30<rate<=40:
                rate = 40
        elif 40<rate<=50:
                rate = 50
        elif 50<rate<=60:
                rate = 60
        elif 60<rate<=70:
                rate = 70
        elif 70<rate<=80:
                rate = 80
        elif rate>80:
                rate = 100
        return rate/100
