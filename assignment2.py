# Them cac thu vien neu can
from functools import reduce
from random import randint,choice,random
from copy import deepcopy as dp
from time import time
import sys

def assign(file_input, file_output):
    
    depot ,amount,shipperNum ,packages,weightMatrix=[None]*5
    
    '''Gán chỉ sổ để truy xuất thể tích, khối lượng gói hàng dễ dàng'''
    WEIGHT  = 4
    VOLUMNE = 3

    # function: ReadInput - version 1.0
    # goal    : read the input from the file with file name store in file_input
    def ReadInput():
        nonlocal depot ,amount,shipperNum ,packages
        file    = open(file_input,"r")
        res     = []
        line    = file.readline()
        while line:
            temp = line.split(" ")
            res += [list(map( lambda x : int(x), temp))]
            line = file.readline()
        
        depot               =(-1,*res[0])
        amount,shipperNum   = res[1]
        packages            = [[i] + ele for i,ele in enumerate(res[2:])]

    # function  : CalProfit - version: 1.0
    # goal      : calculate the profit of a shipper if he move from begin location to the package location
    # Note      : the Progit made by the shipper goes from the depot will be minus by 10
    def CalProfit(begin,package):
        return (
            5  +package[VOLUMNE] + package[WEIGHT]*2
            - (((begin[1]-package[1])**2 +(begin[2]-package[2])**2 )**(1/2))*1/2 
        ) - (10 if begin[0] == -1 else 0)

    # function  : MapNode - version: 1.2
    # goal      : calculate and store every possible profit can be made as a 2 dimension matrix O(n^2)
    def MapNode():
        '''
        Tạo ma trận để lưu vị trí các gói hàng và 
        chi phí từng gói hàng đến các gói hàng còn lại: weightMatrix[i][j]
        '''
        nonlocal depot ,amount,shipperNum ,packages, weightMatrix
        weightMatrix = [[0]*(amount+1) for i in range(amount + 1)]
        for i in range(-1,amount):
            for j in range(-1,amount):
                if i == j or j == -1:
                    weightMatrix[i][j]  = 0
                else:
                    begin               = depot if i == -1 else packages[i]
                    end                 = packages[j]
                    weightMatrix[i][j]  = CalProfit(begin,end)
        
    def GetProfit(pac1,pac2):
        nonlocal depot ,amount,shipperNum ,packages, weightMatrix
        return weightMatrix[pac1][pac2]

    # funtion : Fitness -version : 1.3
    # goal    : return the fitness of a state(a ADN) - O(nlog2(n))`
    def Fitness(costs):
        nonlocal shipperNum
        temp    = costs.copy()
        temp.sort()

        res     = 0
        for i ,val in enumerate(temp):
            res += (-val)*(shipperNum -1 -i) + (val*i)

        return res

    # function : Latter - version: 1.0
    # goal     : get the adjacent package next to the package k in the ADN store in lst
    def Latter(lst,k):
        res = 0
        if k == lst[-1]:
            res = choice(lst)
            while res == k:
                res = choice(lst)
        else:
            res = lst[lst.index(k)+1]
        return res

    # function : Former - version: 1.0
    # goal     : get the adjacent package before of the package k in the ADN store in lst
    def Former(lst,k):
        res = 0
        if k == lst[0]:
            res = choice(lst)
            while res == k:
                res = choice(lst)
        else:
            res = lst[lst.index(k)-1]
        return res

    # function : Part3ADN - version: 1.0
    # goal     : return profits of each shipper
    def Part3ADN(part1,part2):
        nonlocal amount,shipperNum
        
        part3   = []
        profit  = GetProfit(-1,part1[0])

        for i in range(1,amount):
            if i in part2:
                part3   += [profit]
                profit   = GetProfit(-1,part1[i])
            else:
                profit  += GetProfit(part1[i-1],part1[i])

        return part3 + [profit]

    # function : InitPop - version: 1.1
    # goal     : create n random ADN
    def InitPop(n):
        nonlocal amount,shipperNum
        template    = [*range(amount)]
        res         = []
        for _ in range(n):
            breaks = []
            while len(breaks) < shipperNum -1:
                temp = randint(1,amount -1)
                if temp not in breaks:
                    breaks += [temp]

            breaks.sort()
            
            lst     = template.copy()
            shipper = []
            # Costs   = []

            while lst:
                package      = choice(lst)
                lst.remove(package)
                shipper     +=[package]
            

            res += [shipper + breaks + Part3ADN(shipper,breaks)]
        return res 

    # function  : Selection - version: 1.1
    # goal      : return an index of a chosen candidate to be parent for crossover with a certain probability
    # Note      : the better the profit the higher the probability
    def Selection(candidatesFitnesses):
        totalFitness    = sum(candidatesFitnesses)
        compensation    = [totalFitness - i for i in candidatesFitnesses]

        totalFitness    = sum(compensation)
        p               = random()
        temp            = 0

        for i,f in enumerate(compensation):
            temp += f/totalFitness
            if p <= temp:
                return i
        return 0

    # function  : Crossover - version: 1.4
    # goal      : generate an ADN from parents ADNs store in shipperA and shipperB
    # Note      : the algorithm is to try to make the profit of each shipper close to the average Profit of parrents profit
    def Crossover(shipperA, shipperB, forward):
        nonlocal shipperNum,amount
        pA              = shipperA[:amount]
        pB              = shipperB[:amount]

        breaks = []
        avgVal          = sum(shipperA[amount + shipperNum-1:]) + sum(shipperA[amount + shipperNum-1:])
        avgVal          /= 2*shipperNum
       
        pCurr           = randint(0, amount - 1)
        resPackages     = [pCurr]
        resSalemanProf  = []
        salemanProfit   = GetProfit(-1,pCurr)
        assignedShipper = 0

        while len(pA) > 1:
            if forward:
                pANext = Latter(pA,pCurr)
                pBNext = Latter(pB,pCurr)
            else:
                pANext = Former(pA,pCurr)
                pBNext = Former(pB,pCurr)

            pA.remove(pCurr)
            pB.remove(pCurr)

            profitPANext    = GetProfit(pCurr,pANext)
            profitPBNext    = GetProfit(pCurr,pBNext)

            profitPNext     = 0

            if choice([True,False]):
                pCurr       = pANext
                profitPNext = profitPANext
            else:
                pCurr       = pBNext
                profitPNext = profitPBNext

            if (salemanProfit >= avgVal or len(pA) == shipperNum - assignedShipper -1) and (shipperNum - assignedShipper != 1):
                breaks += [len(resPackages)]
                resSalemanProf += [salemanProfit]
                salemanProfit = GetProfit(-1,pCurr)
                assignedShipper += 1
            else:
                salemanProfit+=profitPNext
            resPackages.append(pCurr)

        resSalemanProf += [salemanProfit]
        return resPackages + breaks + resSalemanProf

    # function  : Mutation - version : 1.0
    # goal      : rotate some part of ADN to make the algorithm escape its local minimum
    def Mutation(ADN):
        nonlocal amount,shipperNum

        packagelst      = ADN[:amount]
        idxBreaklst     = ADN[amount:amount + shipperNum -1]
        if (random() <= 0.5):
            # method1
            while True:
                idxBegin    = randint(0, len(packagelst)-1)
                idxEnd      = randint(0, len(packagelst)-1)

                if (idxBegin < idxEnd):
                    break

            newPackagelst   = packagelst[:idxBegin] + \
                (packagelst[idxBegin:(idxEnd+1)])[::-1] + packagelst[(idxEnd+1):]

        else:

            #   method 2
            while True:
                idx1    = randint(0, len(packagelst)-1)
                idx2    = randint(0, len(packagelst)-1)

                if (idx1 > 0) and (idx1 < idx2):
                    break
            
            newPackagelst = packagelst[idx1:(idx2+1)] + packagelst[:idx1] + packagelst[(idx2+1):]

        newIdxBreaklst = []
        begin = 1
        end = amount - shipperNum + 1
        for _ in range(shipperNum -1):
            newIdxBreaklst += [randint(begin,end)]
            begin = newIdxBreaklst[-1]+1
            end += 1
        return newPackagelst + newIdxBreaklst + Part3ADN(newPackagelst,newIdxBreaklst)
    #function   : WriteOutput - version : 1.0
    # goal      : just to write the output to file file_output
    def WriteOutput(ADN):
        nonlocal amount,shipperNum
        part1 = ADN[:amount]
        part2 = ADN[amount:shipperNum + amount - 1]
        
        res = []
        index = 0
        for i in part2:
            res += [part1[index:i]]
            index = i
        res += [part1[index:]]
        
        with open(file_output,"w") as file:
            file.write("\n".join([" ".join([str(j) for j in i]) for i in res]))

    # function  :   MainAlgo -version: 1.5
    # goal      :   run the algorithm
    # Note      :   The problem will be solve by genetic ALgorithm
    #               first the algo will generate N ADN by using InitPop function
    #               Then it will go for a loop of C
    #               for each loop, the algo will make the new N ADN from the ADNs store in 
    #               population using Selection to choose parents and Crossover to generate new ADN
    #               Each ADN has the probability of mutationChance to mutation
    #               update the new ADNs as population
    #               If any of the ADN have fitness equal with 0 or realy close to 0, it would be the result
    #               The algorithm keeps track of the best ADN, if better one founded, the loop will be extend some more time
    def MainAlgo():
        nonlocal amount,shipperNum
        N                       = 15
        population              = InitPop(N)
        mutationChance          = 0.15
        Best,BestFitness,theOne = population[0],Fitness(population[0][amount+shipperNum -1 :]),False
        C                       = 6000
        while C > 0 :
            C -= 1
            if theOne: break
            populationFitness = []
            for candidate in population:
                temp = Fitness(candidate[amount+shipperNum -1 :])
                if temp < BestFitness:
                    C           += 400
                    Best         = candidate
                    BestFitness  = temp

                    if temp - 0 < 1e-7:
                        theOne  = True

                populationFitness += [temp]

            nextGen = []
            for i in range(N):
                parentA = population[Selection(populationFitness)]
                parentB = population[Selection(populationFitness)]
                child   = Crossover(parentA,parentB,choice([True,False]))

                if random() < mutationChance:
                    child = Mutation(child)

                nextGen += [child]

            population = nextGen
        return Best,BestFitness
    

    ReadInput()
    MapNode()
    
    result = MainAlgo()

    WriteOutput(result[0])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise "USAGE: python3 assignment2.py <input.txt> <output.txt>"

    assign(sys.argv[1],sys.argv[2])