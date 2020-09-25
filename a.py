import random, dispy, time, json
import numpy as np


class Creature:
    import random
    ''' 只做每一个染色体进化过程，不涉及交叉、变异等 '''
    def __init__(self, map_size=10, max_step=200, g_loop=200, chrom_size=243):
        self.mapDIM = map_size
        # self.idvNUM = size
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

    # def lookAround(self, pos, nowMap):
    #     # 只能看到上下左右中, pos是(x, y), 但要注意其实 x是纵轴，y是横轴
    #     x, y = pos
    #     return nowMap[x-1][y], nowMap[x+1][y], nowMap[x][y-1], nowMap[x][y+1], nowMap[x][y]

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
        # worst_score = 15 * self.maxStep
        minFIT = abs(min(self.fitness))
        ft_sum = sum(self.fitness) + minFIT * self.size
        # print(ft_sum)

        for i in range(self.size):
            sp[i] = (self.fitness[i] + minFIT) / ft_sum

        # old = 0
        for i in range(self.size):
            sp[i] += sp[i-1]

    def evaluate(self):
        sp = self.selector_probability
        # worst_score = 15 * self.maxStep
        minFIT = min(self.fitness)
        ft_sum = sum(self.fitness) - minFIT*len(self.fitness)
        # print(ft_sum)

        for i in range(self.size):
            sp[i] = (self.fitness[i] - minFIT) / ft_sum

        # old = 0
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
        # i = 0
        # while i < g - self.elitists["age"]:
        #     self.new_individuals[i] = self.elitists["chromosome"][:]
        #     i += 1
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
        # print("".join(map(str, self.individuals[0])))

    def oneGen(self, obj, n, idv, sMX):
        return obj.aLive(n, idv, sMX)

    def fitness_func(self, i):
            # w_p = 0
            # weakness = self.generation_max * 0.01
            # if i < weakness:
            #     w_p = (weakness - i)/weakness
            # if random.random() < w_p:
            #     self.destroy()
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
            # print(i, ": ".join(map(str, self.log[-1])))

    def pipeLOG(self, deltaLog):
        self.logPIPE.write("\t".join(map(str, deltaLog)))
        self.logPIPE.write("\t%s" % self.timeSTR)
        self.logPIPE.write("\n")

    def saveEli(self):
        path = "/home/orin/Learning/Notebook"
        with open("%s/result/%d.res" % (path, int(time.time())), 'w') as f:
            f.write("%s\t%s\n" % (self.timeSTR, json.dumps(self.elitists)))


class Tao_Multi:
    def __init__(self, size=200, chrom_size=243, cp=0.9, mp=0.4, gen_max=1000, island=4):
        self.size = size  # 群体中的个体个数
        self.crossover_probability = cp  # 个体间染色体交叉概率
        # 因为算法中变异率是动态变化的，所以需要给每个island设定变异率
        if island > 0:
            self.mutation_probability = [mp] * island  # 个体变异概率
        else:
            self.mutation_probability = mp
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
        # island=0，意味着经典遗传算法
        self.island = {}
        self.elitists = {
            "chromosome": [],
            "fitness": -15 * 200,
            "age": 0
        }

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
        # worst_score = 15 * self.maxStep
        minFIT = abs(min(self.fitness))
        ft_sum = sum(self.fitness) + minFIT * self.size
        # print(ft_sum)

        for i in range(self.size):
            sp[i] = (self.fitness[i] + minFIT) / ft_sum

        # old = 0
        for i in range(self.size):
            sp[i] += sp[i-1]

    def evaluate(self, island=0):
        # sp = self.selector_probability
        # worst_score = 15 * self.maxStep
        ethnic_s, ethnic_e = self.island[island]
        minFIT = min(self.fitness[ethnic_s: ethnic_e])
        ft_sum = sum(self.fitness[ethnic_s: ethnic_e]) - minFIT*len(self.fitness[ethnic_s:ethnic_e])
        # print(ft_sum)

        for i in range(ethnic_s, ethnic_e):
            self.selector_probability[i] = (self.fitness[i] - minFIT) / ft_sum

        # old = 0
        for i in range(ethnic_s, ethnic_e):
            if i == ethnic_s:
                continue
            else:
                self.selector_probability[i] += self.selector_probability[i-1]

    def destroy(self, s=0, e=10, island=0):
        ethnic_s, ethnic_e = self.island[island]
        minFit = min(self.fitness[ethnic_s:ethnic_e])
        m = 0
        weakness = [
            n for n, val in sorted(list(enumerate(self.fitness))[ethnic_s: ethnic_e], key=lambda x: x[1])
            ][s:e]
        for i in weakness:
            if random.random() < 0.5:
                m += 1
                self.fitness[i] = minFit
        print("!!! Winter Comming: %d Best Individuals are Destroied !!" % m)

    def select(self, island=0):
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

    def mutate(self, chromo, island=0):
        new_chromo = chromo[:]
        p = random.random()
        if p < self.mutation_probability[island]:
            mutate_idx = random.randint(0, self.sample_size-1)
            mutate_val = list(range(self.actions))[random.randint(0, self.actions-1)]
            new_chromo[mutate_idx] = mutate_val
        return new_chromo

    def getElitist(self, age, island=0):
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
            print(
                "$$$ Better Individual Found: age %d, fit %.2f, mutation %.2f." % (
                    age, self.elitists["fitness"], self.mutation_probability
                    )
                    )
        else:
            # 如果达到设定的self.happyGen代不能产生新精英，则判断进入局部最优，摧毁%10的最优个体
            if age - self.elitists["age"][island] > self.happyGen:
                self.destroy(s=int(ethnicScale * 0.10), e=ethnicScale)
                self.elitists["age"][island] = age - (self.happyGen - self.stableGen)
            elif age - self.elitists["age"][island] > self.stableGen:
                # 变异率会突然增加, 但由于变异率在正常情况下不会瞬间下降，所以
                # 需要在后续慢慢下降到设定的值
                if random.random() < (age-self.elitists["age"][island])/(self.happyGen-self.stableGen):
                    self.mutation_probability[island] += 0.05
                    print(
                        "!!! Mutation Warning: %d gen no better individuals, up to %.2f" %
                        (
                            age-self.elitists["age"], self.mutation_probability
                        ))
            else:
                self.mutation_probability[island] -= 0.1
                if self.mutation_probability[island] < self.old_mp:
                    self.mutation_probability[island] = self.old_mp

        if self.mutation_probability[island] > 1:
            self.mutation_probability[island] = 1

    def evolve(self, island):
        ethnic_s, ethnic_e = self.island[island]
        i = ethnic_s + 1
        self.new_individuals[ethnic_s] = self.elitists["chromosome"][island][:]
        # i = 0
        # while i < g - self.elitists["age"]:
        #     self.new_individuals[i] = self.elitists["chromosome"][:]
        #     i += 1
        while True:
            s_chromo1 = self.select(island)
            s_chromo2 = self.select(island)
            (n_chromo1, n_chromo2) = self.cross(
                self.individuals[s_chromo1],
                self.individuals[s_chromo2])
            if random.randint(0, 1) == 0:
                self.new_individuals[i] = self.mutate(n_chromo1, island)
            else:
                self.new_individuals[i] = self.mutate(n_chromo2, island)

            i += 1
            if i >= ethnic_e:
                break
        for i in range(ethnic_s, ethnic_e):
            self.individuals[i] = self.new_individuals[i][:]
        # print("".join(map(str, self.individuals[0])))

    def oneGen(self, obj, n, idv, sMX):
        return obj.aLive(n, idv, sMX)

    def fitness_func(self, i):
            # w_p = 0
            # weakness = self.generation_max * 0.01
            # if i < weakness:
            #     w_p = (weakness - i)/weakness
            # if random.random() < w_p:
            #     self.destroy()
            for n in self.island:
                ethnic_s, ethnic_e = self.island[n]
                self.evaluate(n)
                self.getElitist(i, n)
                # self.log.append([
                #     i,
                #     max(self.fitness),
                #     sum(self.fitness)/len(self.fitness),
                #     min(self.fitness)
                # ])
                self.pipeLOG([
                    i,
                    max(self.fitness),
                    sum(self.fitness)/len(self.fitness),
                    min(self.fitness)
                ])
                self.evolve(i)
            # print(i, ": ".join(map(str, self.log[-1])))

    def pipeLOG(self, deltaLog):
        self.logPIPE.write("\t".join(map(str, deltaLog)))
        self.logPIPE.write("\t%s" % self.timeSTR)
        self.logPIPE.write("\n")

    def saveEli(self):
        path = "/home/orin/Learning/Notebook"
        with open("%s/result/%d.res" % (path, int(time.time())), 'w') as f:
            f.write("%s\t%s\n" % (self.timeSTR, json.dumps(self.elitists)))

    def statSimple(self, fName):
        res = []
        for i in sorted(self.island):
            ethnic_s, ethnic_e = self.island[i]
            r1 = fName(self.fitness[ethnic_s: ethnic_e])
            res.append("isLand %d: %.2f" % (i, r1))
        res.append("Global: %.2f" % fName(self.fitness))
        return res


if __name__ == "__main__":
    n = Tao_Uni(gen_max=1800)

    def disUtil(obj, n, idv, sMX):
        return obj.aLive(n, idv, sMX)

    def disGeneration(map_size=10, max_step=200, g_loop=200, chrom_size=243):
        cluster = dispy.JobCluster(
            disUtil,
            depends=[Creature],
            nodes=["*"],
            secret='Z1207'
            )
        for i in range(n.generation_max):
            jobs = []
            for no, individual in enumerate(n.individuals):
                c = Creature(map_size, max_step, g_loop, chrom_size)
                job = cluster.submit(c, no, individual, n.strategyMX)
                jobs.append(job)

            for job in jobs:
                idx, score = job()
                n.fitness[idx] = score
            n.fitness_func(i)


# 经典遗传算法并行后的测试
# disGeneration(g_loop=400)
# n.logPIPE.close()
# n.saveEli()

m = Tao_Multi()
print(m.elitists)
m.forBorn(1)
print(m.island)
