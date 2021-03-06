import random, dispy, time, json
import numpy as np


class Creature:
    import random
    ''' 只做每一个染色体进化过程，不涉及交叉、变异等 '''
    def __init__(self, map_size=10, max_step=200, g_loop=200, chrom_size=243):
        self.mapDIM = map_size
        self.maxSTEP = max_step
        self.loop = g_loop

        self.act = {
            0: "up",
            1: "down",
            2: "left",
            3: "right",
            4: "rand",
            5: "none",
            6: "pick"
        }
        self.status = {
            0: "none",
            1: "can",
            2: "wall"
        }
        self.pickpunish = {
            0: -1,
            # 应该不会出现人在墙上
            2: -1,
            1: 10
        }

    def show(self):
        print(self.genMAP())

    def genMAP(self):
        tmpmap = []
        # 四边为墙
        upper = [2] * self.mapDIM
        bottom = [2] * self.mapDIM
        tmpmap.append(upper)
        for row in range(self.mapDIM-2):
            tmpmap.append([0] * self.mapDIM)
            tmpmap[row + 1][0] = 2
            for col in range(self.mapDIM-2):
                tmpmap[row + 1][col+1] = self._setItem()
            tmpmap[row + 1][col+2] = 2
        tmpmap.append(bottom)
        return tmpmap

    def lookAround(self, pos, nowMap):
        # 只能看到上下左右中, pos是(x, y), 但要注意其实 x是纵轴，y是横轴
        x, y = pos
        return nowMap[x-1][y], nowMap[x+1][y], nowMap[x][y-1], nowMap[x][y+1], nowMap[x][y]

    def _setItem(self, items=[2, 1, 0], probDis=[0, 0.5, 0.5]):
        # 参数是概率分布数组
        # dispy 需要在函数内import
        import random
        p = random.uniform(0, 1)
        cumulative_probability = 0.0
        for item, item_prob in zip(items, probDis):
            cumulative_probability += item_prob
            if p < cumulative_probability:
                return item

    def showMap(self, tmpmap):
        for i in tmpmap:
            print(" ".join(map(str, i)))

    def _initPos(self):
        import random
        return random.randint(1, self.mapDIM-2), random.randint(1, self.mapDIM-2)

    def score(self, act, pos, nowMap):
        import random
        # 这里实际上是完成了解码和评估两步
        # 解决从染色体到具体得分的过程
        pos = list(pos)
        if act == "rand":
            act = ["up", "down", "left", "right", "pick", "none"][random.randint(0, 5)]
        scoreHash = {
            # “动作”: [x或者y, -1或者1] 用-1乘以这个值完成加减的转换
            "up": [0, 1],
            "down": [0, -1],
            "left": [1, 1],
            "right": [1, -1],
            "none": [0, 0],
            # "rand": [random.randint(0, 1), [-1, 1][random.randint(0, 1)]],
            "pick": [0, 0]
        }
        posIDX = scoreHash[act][0]
        posDelta = scoreHash[act][1]
        pos[posIDX] += -1 * posDelta
        if act == "pick":
            sc = self.pickpunish[nowMap[pos[0]][pos[1]]]
            if nowMap[pos[0]][pos[1]] == 1:
                nowMap[pos[0]][pos[1]] = 0
            return sc, tuple(pos)
        elif nowMap[pos[0]][pos[1]] == 2:
            # 墙壁无法走过去，所以要还原位置，其实很简单只要把乘数-1去掉就好
            pos[posIDX] += posDelta
            return -15, tuple(pos)
        return 0, tuple(pos)

    def aLive(self, no, choromsome, strategyMX):
        # print(no, choromsome, strategyMX)
        # tmpMAP = self.genMAP()
        # pos = self._initPos()
        # score = 0
        score = []
        # 平均一下每个染色体的表现，所以需要self.loop次后取平均值
        for i in range(self.loop):
            score.append(0)
            tmpMAP = self.genMAP()
            pos = self._initPos()
            for m in range(self.maxSTEP):
                situation = self.lookAround(pos, tmpMAP)
                genePOS = strategyMX[situation]
                nowSCORE, pos = self.score(self.act[choromsome[genePOS]], pos, tmpMAP)
                # score += nowSCORE
                score[i] += nowSCORE
        # return no, score/self.loop
        avgScore = sum(sorted(score)[int(self.loop*0.05):int(self.loop*0.95)])/int(self.loop*0.90)
        # avgScore = sum(score)/self.loop
        return no, avgScore


class Tao_Uni:
    def __init__(self, size=200, chrom_size=243, cp=0.9, mp=0.4, gen_max=1000):
        self.size = size  # 群体中的个体个数
        self.crossover_probability = cp  # 个体间染色体交叉概率
        self.mutation_probability = mp  # 个体变异概率
        self.generation_max = gen_max
        self.actions = 7
        self.sample_size = chrom_size
        self.strategyMX = {}
        self.age = 0
        self.log = []
        self.old_mp = mp
        self.happyGen = 100
        self.stableGen = 30
        # 先写入文件，后续考虑改成pipe定向
        self.timeSTR = int(time.time())
        self.logPIPE = open("/home/orin/Learning/Notebook/Genetic.log", 'a', buffering=1)

        # 进化需要的变量声明
        self.individuals = []
        self.fitness = []
        self.selector_probability = []
        self.new_individuals = []
        self.elitists = {
            "chromosome": [],
            "fitness": -15 * 200,
            "age": 0
        }

        self.initVar()

    def initVar(self):
        # 将个体、新个体、适应度、各个体的选择概率等初始化工作单独提出来
        # 由于这些部分不需要分布，所以放在统一函数中，不需要在分布的服务
        # 器上被初始化
        self._genStrategy()
        for i in range(self.size):
            tmpArr = []
            for n in range(self.sample_size):
                tmpArr.append(random.randint(0, self.actions-1))
            self.individuals.append(tmpArr)
            self.fitness.append(0)
            self.selector_probability.append(0)
            self.new_individuals.append([])

    def _genStrategy(self):
        tmp = [0, 0, 0, 0, 0]
        # 3^5=243种枚举策略
        self.strategyMX[tuple(tmp)] = 0
        for i in range(1, self.sample_size):
            tmp = self._addEle(tmp)
            self.strategyMX[tuple(tmp)] = i

    def _addEle(self, tmp):
        m = len(tmp)
        while m >= 0:
            tmp[m-1] += 1
            if tmp[m-1] > 2:
                tmp[m-1] = 0
                m = m - 1
            else:
                return tmp

    def evaluate_bigNum(self):
        sp = self.selector_probability
        minFIT = abs(min(self.fitness))
        ft_sum = sum(self.fitness) + minFIT * self.size

        for i in range(self.size):
            sp[i] = (self.fitness[i] + minFIT) / ft_sum

        for i in range(self.size):
            sp[i] += sp[i-1]

    def evaluate(self):
        sp = self.selector_probability
        minFIT = min(self.fitness)
        ft_sum = sum(self.fitness) - minFIT*len(self.fitness)

        for i in range(self.size):
            sp[i] = (self.fitness[i] - minFIT) / ft_sum

        for i in range(self.size):
            sp[i] += sp[i-1]

    def destroy(self, s=0, e=10):
        minFit = min(self.fitness)
        m = 0
        weakness = [n for n, val in sorted(enumerate(self.fitness), key=lambda x: x[1])][s:e]
        for i in weakness:
            if random.random() < 0.5:
                m += 1
                self.fitness[i] = minFit
        print("!!! Winter Comming: %d Best Individuals are Destroied !!" % m)

    def select(self):
        t = random.random()
        for n, p in enumerate(self.selector_probability):
            if p > t:
                break
        return n

    def showCh(self, chromo):
        return "".join(map(str, chromo))

    def cross(self, chromo1, chromo2):
        p = random.random()
        cross_pos = random.randint(0, self.sample_size-1)
        new_chromo1 = chromo1[:]
        new_chromo2 = chromo2[:]
        if chromo1 != chromo2 and p < self.crossover_probability:
            # 按照书上的交叉，是随机的点进行交换
            new_chromo1, new_chromo2 = chromo1[: cross_pos], chromo2[: cross_pos]
            new_chromo1.extend(chromo2[cross_pos:])
            new_chromo2.extend(chromo1[cross_pos:])
        return new_chromo1, new_chromo2

    def mutate(self, chromo):
        new_chromo = chromo[:]
        p = random.random()
        if p < self.mutation_probability:
            mutate_idx = random.randint(0, self.sample_size-1)
            mutate_val = list(range(self.actions))[random.randint(0, self.actions-1)]
            new_chromo[mutate_idx] = mutate_val
        return new_chromo

    def getElitist(self, age):
        bestIndividual = [[idx, fit] for idx, fit in sorted(
            enumerate(self.fitness), key=lambda x: x[1], reverse=True
        )][0]
        if self.elitists["fitness"] < self.fitness[bestIndividual[0]]:
            self.elitists["chromosome"] = []
            self.elitists["age"] = age
            self.elitists["chromosome"].extend(self.individuals[bestIndividual[0]])
            self.elitists["fitness"] = self.fitness[bestIndividual[0]]
            print("$$$ Better Individual Found: age %d, fit %.2f, mutation %.2f." % (age, self.elitists["fitness"], self.mutation_probability))
        else:
            # 如果达到设定的self.happyGen代不能产生新精英，则判断进入局部最优，摧毁%10的最优个体
            if age - self.elitists["age"] > self.happyGen:
                self.destroy(s=int(len(self.fitness) * 0.10), e=len(self.fitness))
                self.elitists["age"] = age - 70
            elif age - self.elitists["age"] > self.stableGen:
                # 变异率会突然增加, 但由于变异率在正常情况下不会瞬间下降，所以
                # 需要在后续慢慢下降到设定的值
                if random.random() < (age-self.elitists["age"])/(self.happyGen-self.stableGen):
                    self.mutation_probability += 0.05
                    print("!!! Mutation Warning: %d gen no better individuals, up to %.2f" % (age-self.elitists["age"], self.mutation_probability))
            else:
                self.mutation_probability -= 0.1
                if self.mutation_probability < self.old_mp:
                    self.mutation_probability = self.old_mp

        if self.mutation_probability > 1:
            self.mutation_probability = 1

    def evolve(self, g):
        i = 1
        self.new_individuals[0] = self.elitists["chromosome"][:]
        while True:
            s_chromo1 = self.select()
            s_chromo2 = self.select()
            (n_chromo1, n_chromo2) = self.cross(
                self.individuals[s_chromo1],
                self.individuals[s_chromo2])
            if random.randint(0, 1) == 0:
                self.new_individuals[i] = self.mutate(n_chromo1)
            else:
                self.new_individuals[i] = self.mutate(n_chromo2)

            i += 1
            if i >= self.size:
                break
        for i in range(self.size):
            self.individuals[i] = self.new_individuals[i][:]

    def oneGen(self, obj, n, idv, sMX):
        return obj.aLive(n, idv, sMX)

    def fitness_func(self, i):
            self.evaluate()
            self.getElitist(i)
            self.log.append([
                i,
                max(self.fitness),
                sum(self.fitness)/len(self.fitness),
                min(self.fitness)
            ])
            self.pipeLOG([
                i,
                max(self.fitness),
                sum(self.fitness)/len(self.fitness),
                min(self.fitness)
            ])
            self.evolve(i)

    def pipeLOG(self, deltaLog):
        self.logPIPE.write("\t".join(map(str, deltaLog)))
        self.logPIPE.write("\t%s" % self.timeSTR)
        self.logPIPE.write("\n")

    def saveEli(self):
        path = "/home/orin/Learning/Notebook"
        with open("%s/result/%d.res" % (path, int(time.time())), 'w') as f:
            f.write("%s\t%s\n" % (self.timeSTR, json.dumps(self.elitists)))


class Tao_Multi:
    def __init__(self, size=200, chrom_size=243, cp=0.9, mp=0.4, tp=0.1, gen_max=1000, island=4, diff=False, dp=0.5):
        self.size = size  # 群体中的个体个数
        self.destroy_probability = dp
        self.crossover_probability = cp  # 个体间染色体交叉概率
        self.maxMP = mp
        initMP = self.randomMP(island, mp)
        print("Init Mutation Probability:", initMP)
        self.stable_p = {"mp": initMP[:], "tp": tp}
        # 因为算法中变异率是动态变化的，所以需要给每个island设定变异率
        if island > 0:
            self.mutation_probability = initMP[:]  # 个体变异概率
        else:
            self.mutation_probability = mp
        self.generation_max = gen_max
        self.dukeTown = random.randint(0, island-1)
        print("Saddly, island %d is chosen" % self.dukeTown)
        self.actions = 7
        self.sample_size = chrom_size
        self.strategyMX = {}
        self.age = 0
        self.log = []
        # self.old_mp = mp
        self.happyGen = 100
        self.stableGen = 30
        # 先写入文件，后续考虑改成pipe定向
        self.timeSTR = int(time.time())
        self.logPIPE = open("/home/orin/Learning/Notebook/Genetic.mi.log", 'a', buffering=1)

        # 进化需要的变量声明
        self.individuals = []
        self.fitness = []
        self.selector_probability = []
        self.new_individuals = []
        # island=0，意味着经典遗传算法
        self.island = {}
        self.elitists = {
            "chromosome": [[]] * island,
            "fitness": [-15 * 200]*4,
            "age": [0] * 4
        }
        self.trans = [0] * island
        self.tp = [tp] * island
        # 不同交叉规则的开关
        self.varCross = [False] * island
        self.afterDestory = [False] * island
        self.diff = {
            1: self.uniformCross,
            0: self.cross
        }
        if diff:
            # 默认选择一半岛用更多样的交叉规则
            self.varCross[0] = 1
            self.varCross[2] = 1

        self.initVar()
        self.forBorn(island)

    def initVar(self):
        # 将个体、新个体、适应度、各个体的选择概率等初始化工作单独提出来
        # 由于这些部分不需要分布，所以放在统一函数中，不需要在分布的服务
        # 器上被初始化
        self._genStrategy()
        for i in range(self.size):
            tmpArr = []
            for n in range(self.sample_size):
                tmpArr.append(random.randint(0, self.actions-1))
            self.individuals.append(tmpArr)
            self.fitness.append(0)
            self.selector_probability.append(0)
            self.new_individuals.append([])

    def randomMP(self, island, mp):
        tmp = []
        halfIdv = int(island/2)
        # 最小变异率10%，实际上mp是最大的变异可能
        tmp.extend([random.uniform(0.1, mp/2) for i in range(halfIdv)])
        tmp.extend([random.uniform(mp/2, mp) for i in range(island-halfIdv)])
        random.shuffle(tmp)
        return tmp

    def forBorn(self, island):
        # 直接按照岛屿数量将所有个体顺序切分，除不尽的直接在最后一个岛屿补齐
        totaLen = len(self.fitness)
        step = int(totaLen/island)
        if island < 1:
            self.island[island] = [0, totaLen]
            return True
        else:
            for i in range(island):
                self.island[i] = [int(i*step), int((i+1)*step)]
            if self.island[i][1] < totaLen:
                self.island[i][1] = totaLen

    def _genStrategy(self):
        tmp = [0, 0, 0, 0, 0]
        # 3^5=243种枚举策略
        self.strategyMX[tuple(tmp)] = 0
        for i in range(1, self.sample_size):
            tmp = self._addEle(tmp)
            self.strategyMX[tuple(tmp)] = i

    def _addEle(self, tmp):
        m = len(tmp)
        while m >= 0:
            tmp[m-1] += 1
            if tmp[m-1] > 2:
                tmp[m-1] = 0
                m = m - 1
            else:
                return tmp

    def evaluate_bigNum(self):
        sp = self.selector_probability
        minFIT = abs(min(self.fitness))
        ft_sum = sum(self.fitness) + minFIT * self.size

        for i in range(self.size):
            sp[i] = (self.fitness[i] + minFIT) / ft_sum

        # old = 0
        for i in range(self.size):
            sp[i] += sp[i-1]

    def evaluate(self, island=0):
        ethnic_s, ethnic_e = self.island[island]
        minFIT = min(self.fitness[ethnic_s: ethnic_e])
        ft_sum = sum(self.fitness[ethnic_s: ethnic_e]) - minFIT*len(self.fitness[ethnic_s:ethnic_e])

        for i in range(ethnic_s, ethnic_e):
            self.selector_probability[i] = (self.fitness[i] - minFIT) / ft_sum

        for i in range(ethnic_s, ethnic_e):
            if i == ethnic_s:
                continue
            else:
                self.selector_probability[i] += self.selector_probability[i-1]

    def destroy(self, s=0, e=10, island=0):
        island = self.dukeTown
        ethnic_s, ethnic_e = self.island[island]
        minFit = min(self.fitness[ethnic_s:ethnic_e])
        m = 0
        weakness = [
            n for n, val in sorted(list(enumerate(self.fitness))[ethnic_s: ethnic_e], key=lambda x: x[1])
            ][s:e]
        for i in weakness:
            if random.random() < self.destroy_probability:
                m += 1
                self.fitness[i] = minFit
        print('\033[5;37m', end='')
        print("!!! Winter Comming to Island %d: %d Best Individuals are Destroied !! Mutation ReLoaded." % (island, m), end='')
        print('\033[0m')
        # 让被destroy的island自主发展一段时间
        self.mutation_probability[island] = self.stable_p["mp"][island]
        self.tp[island] = self.stable_p["tp"]
        self.afterDestory[island] = True

    def select(self, island):
        ethnic_s, ethnic_e = self.island[island]
        t = random.random()
        for n, p in list(enumerate(self.selector_probability))[ethnic_s:ethnic_e]:
            if p > t:
                break
        return n

    def showCh(self, chromo):
        return "".join(map(str, chromo))

    def cross(self, chromo1, chromo2):
        p = random.random()
        cross_pos = random.randint(0, self.sample_size-1)
        new_chromo1 = chromo1[:]
        new_chromo2 = chromo2[:]
        if chromo1 != chromo2 and p < self.crossover_probability:
            # 按照书上的交叉，是随机的点进行交换
            new_chromo1, new_chromo2 = chromo1[: cross_pos], chromo2[: cross_pos]
            new_chromo1.extend(chromo2[cross_pos:])
            new_chromo2.extend(chromo1[cross_pos:])
        return new_chromo1, new_chromo2

    def uniformCross(self, chromo1, chromo2):
        p = random.random()
        new_chromo1 = chromo1[:]
        new_chromo2 = chromo2[:]
        if p < self.crossover_probability:
            for i in range(len(chromo1)):
                p_c = random.random()
                if p_c < 0.5:
                    new_chromo1[i], new_chromo2[i] = new_chromo2[i], new_chromo1[i]
        return new_chromo1, new_chromo2

    def mutate(self, chromo, island):
        new_chromo = chromo[:]
        p = random.random()
        if p < self.mutation_probability[island]:
            mutate_idx = random.randint(0, self.sample_size-1)
            mutate_val = list(range(self.actions))[random.randint(0, self.actions-1)]
            new_chromo[mutate_idx] = mutate_val
        return new_chromo

    def getElitist(self, age, island):
        ethnic_s, ethnic_e = self.island[island]
        ethnicScale = len(self.fitness[ethnic_s: ethnic_e])
        bestIndividual = [[idx, fit] for idx, fit in sorted(
            list(enumerate(self.fitness))[ethnic_s: ethnic_e], key=lambda x: x[1], reverse=True
        )][0]
        if self.elitists["fitness"][island] < self.fitness[bestIndividual[0]]:
            self.elitists["chromosome"][island] = []
            self.elitists["age"][island] = age
            self.elitists["chromosome"][island].extend(self.individuals[bestIndividual[0]])
            self.elitists["fitness"][island] = self.fitness[bestIndividual[0]]
            self.tp[island] = self.stable_p["tp"]
            self.mutation_probability[island] = self.stable_p["mp"][island]
            print('\033[1;33m', end='')
            print(
                "$$$ Island %d Better Individual Found: age %d, fit %.2f, mutation %.2f." % (
                    island, age, self.elitists["fitness"][island], self.mutation_probability[island]
                    ), end=''
                    )
            print('\033[0m')
        else:
            # 如果达到设定的self.happyGen代不能产生新精英，则判断进入局部最优，摧毁%10的最优个体
            if age - self.elitists["age"][island] > self.happyGen:
                self.destroy(s=int(ethnicScale * 0.10), e=ethnicScale, island=island)
                self.elitists["age"][island] = age - (self.happyGen - self.stableGen)
                self.trans[island] = [n for n, v in sorted(enumerate(self.elitists["fitness"]), key=lambda x: x[1]) if n != island][0]
            elif age - self.elitists["age"][island] > self.stableGen:
                # 变异率会突然增加, 但由于变异率在正常情况下不会瞬间下降，所以
                # 需要在后续慢慢下降到设定的值
                if random.random() < (age-self.elitists["age"][island])/(self.happyGen-self.stableGen):
                    # 步数有问题时，很难稳定，所以暂停变异率增加
                    self.mutation_probability[island] += 0.05
                    self.tp[island] += 0.05
                    print('\033[1;31m', end='')
                    print(
                        "!!! Island %d Mutation Warning: %d gen no better individuals, up to %.2f, trans %.2f" %
                        (
                            island, age-self.elitists["age"][island], self.mutation_probability[island], self.tp[island]
                        ), end='')
                    print('\033[0m')
            else:
                # self.mutation_probability[island] -= 0.1
                if self.mutation_probability[island] < self.stable_p["mp"][island]:
                    self.mutation_probability[island] = self.stable_p["mp"][island]

        if self.mutation_probability[island] > self.maxMP/2+0.5:
            # 变异率提升到 1 虽然可以提高突破局部最优的可能，但是
            # 并不利于稳定性状的保持。所以将变异率的最大值锁定在
            # 预设变异率，这样预设变异率其实就成为了最大变异率了。
            # (1 - self.maxMP)/2 + self.maxMP
            self.mutation_probability[island] = self.maxMP/2+0.5
        # if self.mutation_probability[island] > 1:
        #     self.mutation_probability[island] = 1

    def islandTrans(self, age, island):
        # 按照1->2->3->4->1，所以定一个island长度的列表
        # 0 为不传递，1为传递
        acc_p = self.simAnneal(age-self.elitists["age"][island])
        if self.trans[island]:
            return self.elitists["chromosome"][self.trans[island]]
        else:
            if self.elitists["chromosome"][island] < self.elitists["chromosome"][island-1]:
                if random.random() < (1-acc_p):
                    return self.elitists["chromosome"][island-1]
            else:
                if random.random() < acc_p:
                    return self.elitists["chromosome"][island-1]
        return False

    def simAnneal(self, age):
        return 0.5*(0.8**int(age/100))

    # def elitistFunc(self, i, island):
    #     if self.afterDestory[island]:
    #         for m in range(len(self.tp)):
    #         self.new_individuals[i] = self.elitists["chromosome"][island][:]

    def evolve(self, age, island):
        ethnic_s, ethnic_e = self.island[island]
        i = ethnic_s
        self.new_individuals[i] = self.elitists["chromosome"][island][:]
        i += 1
        trans = self.islandTrans(age, island)
        if trans and random.random() < self.tp[island]:
            print("!!! Trans: %d" % island)
            self.new_individuals[i] = trans[:]
            if self.trans[island]:
                self.trans[island] = 0
            i += 1
        if self.tp[island] < self.stable_p["tp"]: self.tp[island] = self.stable_p["tp"]
        if self.tp[island] > 1 : self.tp[island] = 1
        while True:
            s_chromo1 = self.select(island)
            s_chromo2 = self.select(island)
            (n_chromo1, n_chromo2) = self.diff[self.varCross[island]](
                self.individuals[s_chromo1],
                self.individuals[s_chromo2]
            )
            if random.randint(0, 1) == 0:
                self.new_individuals[i] = self.mutate(n_chromo1, island)
            else:
                self.new_individuals[i] = self.mutate(n_chromo2, island)

            i += 1
            if i >= ethnic_e:
                break
        for i in range(ethnic_s, ethnic_e):
            self.individuals[i] = self.new_individuals[i][:]

    def oneGen(self, obj, n, idv, sMX):
        return obj.aLive(n, idv, sMX)

    def fitness_func(self, i):
            print("%s %d %s" % ("="*15, i, "="*15))
            for n in self.island:
                ethnic_s, ethnic_e = self.island[n]
                self.evaluate(n)
                self.getElitist(i, n)
                self.evolve(i, n)
            self.pipeLOG([
                i,
                self.statSimple(max),
                self.statSimple(np.mean),
                self.statSimple(min)
            ])

    def pipeLOG(self, deltaLog):
        self.logPIPE.write("\t".join(map(str, deltaLog)))
        self.logPIPE.write("\t%s" % self.timeSTR)
        self.logPIPE.write("\n")

    def saveEli(self):
        path = "/home/orin/Learning/Notebook"
        with open("%s/result/%d.mi.res" % (path, int(time.time())), 'w') as f:
            f.write("Cons\t%s\t%s\n" % (self.dukeTown, json.dumps(self.stable_p)))
            f.write("Res\t%s\t%s\n" % (self.timeSTR, json.dumps(self.elitists)))

    def statSimple(self, fName):
        res = []
        res.append("%.2f" % fName(self.fitness))
        for i in sorted(self.island):
            ethnic_s, ethnic_e = self.island[i]
            r1 = fName(self.fitness[ethnic_s: ethnic_e])
            res.append("%.2f" % r1)
        return res


if __name__ == "__main__":
    # 单线程版
    # n = Tao_Uni(gen_max=1800)

    # def disUtil(obj, n, idv, sMX):
    #     print(n, idv, sMX)
    #     return obj.aLive(n, idv, sMX)

    # def disGeneration(map_size=12, max_step=200, g_loop=200, chrom_size=243):
    #     cluster = dispy.JobCluster(
    #         disUtil,
    #         depends=[Creature],
    #         nodes=["*"],
    #         secret='Z1207'
    #         )
    #     for i in range(n.generation_max):
    #         jobs = []
    #         for no, individual in enumerate(n.individuals):
    #             c = Creature(map_size, max_step, g_loop, chrom_size)
    #             job = cluster.submit(c, no, individual, n.strategyMX)
    #             jobs.append(job)

    #         for job in jobs:
    #             idx, score = job()
    #             n.fitness[idx] = score
    #         n.fitness_func(i)
    ####################################

    # 经典遗传算法并行后的测试
    # disGeneration(g_loop=400)
    # n.logPIPE.close()
    # n.saveEli()

    # 多机分布版本
    m = Tao_Multi(gen_max=3000, diff=True, dp=0.8, tp=0.05)

    def disUtil_Multi(obj, n, idv, sMX):
            return obj.aLive(n, idv, sMX)

    def disGeneration_Multi(map_size=12, max_step=200, g_loop=200, chrom_size=243, island=4):
            cluster = dispy.JobCluster(
                disUtil_Multi,
                depends=[Creature],
                nodes=["*"],
                secret='Z1207'
                )
            for i in range(m.generation_max):
                jobs = []
                for no, individual in enumerate(m.individuals):
                    c = Creature(map_size, max_step, g_loop, chrom_size)
                    job = cluster.submit(c, no, individual, m.strategyMX)
                    jobs.append(job)

                for job in jobs:
                    idx, score = job()
                    m.fitness[idx] = score
                m.fitness_func(i)
                if max(m.fitness) > (map_size-2)*(map_size-2)*10/2-10:
                    break

    disGeneration_Multi(g_loop=500)
    m.logPIPE.close()
    m.saveEli()
