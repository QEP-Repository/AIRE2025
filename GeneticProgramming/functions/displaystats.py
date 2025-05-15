def displaystats(gp):
    
    if gp['state']['count'] % 10 == 0: 
        print(f'generation: {gp["state"]["count"]}')
        print(f'best_fitness: {gp["state"]["best"]["fitness"]}')
        print(f'complexity: {gp["state"]["best"]["complexity"]}')
        print(f'indiv: {gp["state"]["best"]["individual"]}')
        print('')
              
    return gp