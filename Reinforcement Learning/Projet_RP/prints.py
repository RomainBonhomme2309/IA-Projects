### Environnements
from environnements.hangar_world import HangarWorldMDP
from environnements.entrepot_world import EntrepotWorldMDP
from environnements.garage_world import GarageWorldMDP

if __name__ == "__main__":
    hangar = HangarWorldMDP(4, 4, 1, 2)
    hangar.print_board()
    
    entrepot = EntrepotWorldMDP(4, 4, 1, 10)
    entrepot.print_board()
    
    garage = GarageWorldMDP(4, 4, 1, 2)
    garage.print_board()