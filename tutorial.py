import pygame
from pong import Game
import neat
import os
import pickle

class PongGame:
    def __init__(self, window, width, height):
        self.game = Game(window, width, height)
        #getting information for AI
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.ball = self.game.ball

    def test_ai(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        run = True
        clock = pygame.time.Clock()
        
        while run:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break
            
            #human player
            keys = pygame.key.get_pressed()
            
            if keys[pygame.K_w]:
                self.game.move_paddle(left = True, up = True)
            if keys[pygame.K_s]:
                self.game.move_paddle(left = True, up = False)
            

            #AI
            output = net.activate((self.right_paddle.y, self.ball.y, abs(self.right_paddle.x - self.ball.x)))
            decision = output.index(max(output))
            
            if decision == 0:
                pass
            elif decision == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)

            game_info = self.game.loop()
            self.game.draw()
            pygame.display.update()

        pygame.quit()

    def train_ai(self, genome1, genome2, config):
        #initializing neural networks
        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)
        
        run = True
        # clock = pygame.time.Clock()
        
        while run:
            # clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                    # return False           
            
            #setting the inputs and getting the outputs of each neural network
            output1 = net1.activate((self.left_paddle.y, self.ball.y, abs(self.left_paddle.x - self.ball.x)))
            decision1 = output1.index(max(output1)) #gets the max value of the output and return its index
            #[stay, up, down]

            if decision1 == 0:
                pass
            elif decision1 == 1:
                self.game.move_paddle(left=True, up=True)
            else:
                self.game.move_paddle(left=True, up=False)
            
            #do the same to the other paddle
            output2 = net2.activate((self.right_paddle.y, self.ball.y, abs(self.right_paddle.x - self.ball.x)))
            decision2 = output2.index(max(output2))
            
            if decision2 == 0:
                pass
            elif decision2 == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)
            

            
            game_info = self.game.loop()
            self.game.draw(draw_score=False, draw_hits=True)
            pygame.display.update()

            if game_info.left_score >= 1 or game_info.right_score >=1 or game_info.left_hits > 50:
                self.calculate_fitness(genome1, genome2, game_info)
                break
    
    def calculate_fitness(self, genome1, genome2, game_info):
        genome1.fitness += game_info.left_hits
        genome2.fitness += game_info.right_hits


def eval_genomes(genomes, eval): #fitness function
    # we have a problem: we can't evaluate well all the AIs 
    # if they don't train against the same opponent
    # so we're going to do the following: each AI will compete against all others AISs
    # its fitness will be the sum of all rounds it has played
    width, height = 700, 500
    window = pygame.display.set_mode((width, height))

    #playing against each other
    for i, (genome_id1, genome1) in enumerate(genomes):
        if i == len(genomes) - 1:
            break
        genome1.fitness = 0
        
        for genome_id2, genome2 in genomes[i+1:]:
            genome2.fitness = 0 if genome2.fitness is None else genome2.fitness #if the fitness doesn't exist, it will be 0
            game = PongGame(window, width, height)

            force_quit = game.train_ai(genome1, genome2, config)

def run_neat(config):
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-10') #reloads a population checkpoint
    # p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1)) #saves a checkpoint after a number (here: 1) of generations

    winner = p.run(eval_genomes, 5)
    #saving the winner
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)

def test_ai(config):
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)
    
    width, height = 700, 500
    window = pygame.display.set_mode((width, height))

    game = PongGame(window, width, height)
    game.test_ai(winner, config)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    # run_neat(config)
    test_ai(config)