'''
Created on Nov. 2, 2012

@author: hitokazu
'''

import sys
import time
from bzrc import BZRC, Command, Answer

#from grid_filter_gl import *
#
#import numpy as np

from agent import Agent

#class Agent(object):
#    """Class handles all command and control logic for a teams tanks."""
#
#    def __init__(self, bzrc):
#        self.debug = False
#        self.bzrc = bzrc
#        self.constants = self.bzrc.get_constants()
#        self.attractive_s = 10
#        self.attractive_alpha = 0.7
#        self.grid_size = 32
#        self.grid_step = 0.5
#        self.upper_theshold = 0.95
#        self.lower_theshold = 0.05
#        self.speed = 1
#        self.time_threshold = 5
#        self.found = True
#        self.fire = False
#        self.initial_prior = 0.75
#        self.infinity = float("inf")
#        self.constants = self.bzrc.get_constants()
#        self.stuck = False
#        self.minimum_distance = 1
#        self.colors = ['red', 'blue', 'green', 'purple']
#        self.prev = []
#        # constants values:
#        #{'shotspeed': '100', 'tankalive': 'alive', 'truepositive': '0.97', 'worldsize': '800', 
#        # 'explodetime': '5', 'truenegative': '0.9', 'shotrange': '350', 'flagradius': '2.5', 
#        # 'tankdead': 'dead', 'tankspeed': '25', 'shotradius': '0.5', 
#        # 'tankangvel': '0.785398163397', 'linearaccel': '0.5', 'team': 'red', 
#        # 'tankradius': '4.32', 'angularaccel': '0.5', 'tankwidth': '2.8', 'tanklength': '6'}
#
#        self.commands = []
#        self.worldsize = int(self.constants['worldsize']) # the values are not integers; you need to cast 
#        init_window(self.worldsize, self.worldsize)
#        s = (self.worldsize, self.worldsize)
#        self.priors = np.ones(s) * self.initial_prior 
#        update_grid(self.priors)
#        draw_grid()
#
#        mytanks, othertanks, flags, shots = self.bzrc.get_lots_o_stuff()
#        self.numtank = len(mytanks)
#        self.set_initial_prev()
#        #print "tank index: %d" % mytanks[0].index        
#        self.set_attractive_points(mytanks[0])
#         
#    def set_initial_prev(self):
#        for i in xrange(self.numtank):
#            self.prev.append((0,0))
#         
#    def tick(self, time_diff):
#        """Some time has passed; decide what to do next."""
#        mytanks, othertanks, flags, shots = self.bzrc.get_lots_o_stuff()
#        self.mytanks = mytanks
#        self.othertanks = othertanks
#        self.shots = shots
#
#        self.commands = []
#
#        for tank in self.mytanks:
#            self.prev_t = time.time()
#            self.update_probabilities(tank)
#            self.prev[tank.index] = (tank.x, tank.y)
#        
#        results = self.bzrc.do_commands(self.commands)
#
#    def set_attractive_points(self, tank):
#        """ set the positions to be visited"""
#        self.grids = []
#        size = self.bzrc.get_occgrid(tank.index)[2]
#        occgrid_width = max(size[0], size[1])
#        self.occgrid_width = occgrid_width         
#
#        #self.grids.append((0,0))
#
#        #reverse = False
#
#        for y in xrange(int(occgrid_width/self.grid_size), self.worldsize, occgrid_width/int(self.grid_size*self.grid_step)):
#            grids = []
#            for x in xrange(int(occgrid_width/self.grid_size), self.worldsize, occgrid_width/int(self.grid_size*self.grid_step)):
#                grids.append((x-400, y-400))
#            self.grids.append(grids)
#            
##            if reverse == False:
##                reverse = True
##            else:
##                reverse = False
##                grids.reverse()
##            for grid in grids:
##                self.grids.append(grid)
#        
#        #print "%d x %d" % (len(self.grids), len(self.grids[0])) 
#        self.grids = self.separate_gridlist()
#        if len(self.grids) > 1:
#            for i in xrange(len(self.grids)):
#                if i+1 > int(len(self.grids)*0.5):
#                    #print "%d is reversed" % i
#                    self.grids[i].reverse()
#        #print len(self.grids)
#
#        self.flags = []
#
#        for grids in self.grids: # iterate over flag list separated by # of tanks
#            #self.flags.append(self.create_flag(grid[0], grid[1]))
#            #self.flags.append(self.set_flags(grids))
#            self.flags.append(self.set_separated_flags(grids))
#
#        if self.debug == True:
#            print "%d x %d x %d" % (len(self.flags), len(self.flags[0]), len(self.flags[0][0]))
#
#    def separate_gridlist(self):
#        """ create 3-D grid list """
#        separeted_grids = []
#        start = 0
#        step = len(self.grids)/self.numtank
#        
#        for i in xrange(self.numtank):
#            grids = []
#            grids = self.grids[start:step]
#            separeted_grids.append(grids)
#            start = step
#            step += step
#
#        return separeted_grids
#
#    def create_flag(self, x, y):
#        """ create a new flag """
#        flag = Answer()
#        for color in self.colors:
#            if color != self.constants['team']:
#                break
#        flag.color = color
#        flag.poss_color = color
#        flag.x = float(x)
#        flag.y = float(y)
#        
#        return flag
#        
#    def set_flags(self, grids):
#        """ set a list of new flags """
#        flags = []
#        for grid in grids:
#            flag = self.create_flag(grid[0], grid[1])
#            flags.append(flag)
#        
#        return flags
#    
#    def set_separated_flags(self, grids):
#        """ set flag lists separeted by # of tanks"""
#        flaglist = []
#        for gridlist in grids:
#            flaglist.append(self.set_flags(gridlist))
#        return flaglist
#
#    def update_probabilities(self, tank):
#        """ update grid prior probabilities of each grid with posterior probabilities  """
#        grid = self.bzrc.get_occgrid(tank.index)
#        #print grid
#        #print grid[0], len(grid[1]), len(grid[1][0]), grid[1][0]
#        center_x = (grid[0][1])+self.worldsize/2
#        center_y = (grid[0][0])+self.worldsize/2
#        rows = len(grid[1])
#        cols = len(grid[1][0])
#
#        #print center_x, center_y, rows, cols
#
#        for row in xrange(rows):
#            for col in xrange(cols):
#                sensor_x = center_x + col#- int(rows/2) + col
#                sensor_y = center_y + row#- int(cols/2) + row
#                if sensor_x < 0 or sensor_y < 0 or sensor_x > 799 or sensor_y > 799:
#                    continue
#                else:
#                    true_positive = float(self.constants['truepositive'])
#                    true_negative = float(self.constants['truenegative'])
#                    #print sensor_x, sensor_y
#                    if grid[1][row][col] == 1:
#                        # occupied (unnormalized posterior) = P(O=1|S=1) * P(S=1) [true positive * prior]
#                        occupied = true_positive * self.priors[sensor_x][sensor_y]
#                        # unoccupied (unnormalized posterior) = P(O=1|S=0) * P(S=0) [false alarm * prior]
#                        unoccupied = (1 - true_positive) * (1 - self.priors[sensor_x][sensor_y])
#                        # normalized posterior P(O=1|S=1); this is assigned as the next prior at the current position
#                        #self.priors[sensor_x][sensor_y] = self.set_probablity(occupied / (occupied + unoccupied), sensor_x, sensor_y)
#                        self.priors[sensor_x][sensor_y] = occupied / (occupied + unoccupied)
#                    else:
#                        # occupied (unnormalized posterior) = P(O=0|S=1) * P(S=1) [missed detection * prior]
#                        occupied =  (1 - true_positive) * self.priors[sensor_x][sensor_y]
#                        # unoccupied (unnormalized posterior) = P(O=0|S=0) * P(S=0) [true negative * prior]
#                        unoccupied = true_negative * (1 - self.priors[sensor_x][sensor_y])
#                        # normalized posterior P(O=1|S=1); this is assigned as the next prior at the current position
#                        #self.priors[sensor_x][sensor_y] = self.set_probablity(occupied / (occupied + unoccupied), sensor_x, sensor_y)
#                        self.priors[sensor_x][sensor_y] = occupied / (occupied + unoccupied)
#                        
#                    #self.priors[sensor_x][sensor_y] = 1.0
#        #self.priors[(grid[0][1])+400][grid[0][0]+400] = 1.0
#        update_grid(self.priors)
#        draw_grid()
#        self.eliminate_flag(tank)
#        self.get_direction(tank)                    
#
#    def set_probablity(self, p, x, y):
#        if p > self.upper_theshold:
#            return 1
#        if p < self.lower_theshold:
#            return 0
#        
#        return p
#        #self.check_surrounding_cells(p, x, y)
#
#    def check_surrounding_cells(self, p, x, y):
#        ys = [y-1, y, y+1]
#        xs = [x-1, x, x+1]
#        
#        ones = 0
#        zeros = 0
#        
#        for x in xs:
#            if x > 800 or x < 0:
#                continue
#            else:
#                for y in ys:
#                    if y > 800 or y < 0:
#                        continue
#                    else:
#                        if self.priors[x][y] >= self.upper_theshold:
#                            ones += 1
#                            continue
#                        if self.priors[x][y] <= self.lower_theshold:
#                            zeros += 1 
#                
#        if ones > zeros:
#            p = 1
#        else:
#            p = 0
#        
#
#    def get_distance(self, tank):
#        """ get the distance from the previous position"""
#        #print self.prev
#        d = math.sqrt((tank.x - self.prev[tank.index][0])**2 + (tank.y - self.prev[tank.index][1])**2)
#        #d = (tank.x - self.prev[tank.index][0])**2 + (tank.y - self.prev[tank.index][1])**2
#        return d
#    
#    def is_stuck(self, tank):
#        """ check if the tank is moving """
#        d = self.get_distance(tank)
#        if self.debug == True:
#            print "Distance: %.2f" % d
#        if d < self.minimum_distance:
#            return True
#        else:
#            return False
# 
#    def eliminate_flag(self, tank):
#        """ remove visited grids (attractive points) """
#        found = False
#        for k in xrange(len(self.grids)): # grid chunk divided by # of tanks
#            for j in xrange(len(self.grids[k])): # rows in self.grids
#                for i in xrange(len(self.grids[k][j])): # cols in self.grids
#                    #print "grid: %d, %d \t prior: %.2f" % (self.grids[j][i][0] + 400, self.grids[j][i][1] + 400, self.priors[self.grids[j][i][0] + 400][self.grids[j][i][1] + 400])
#                    #print "prior: %.2f" % self.priors[self.grids[j][i][0] + 400][self.grids[j][i][1] + 400]
#                    if self.flags[k][j][i].poss_color != self.constants['team']:
#                        d = (tank.x - self.grids[k][j][i][0])**2 + (tank.y - self.grids[k][j][i][1])**2
#                        if d < 0.9 * ((self.occgrid_width/2) ** 2):
#                        #if self.priors[position[0]][position[1]] != self.initial_prior:
#                            found = True
#                            if self.debug == True: 
#                                position = (self.grids[k][j][i][0] + 400, self.grids[k][j][i][1] + 400)
#                                print "updated (%d, %d)" % (position[0]-400, position[1]-400)
#                            self.flags[k][j][i].poss_color = self.constants['team']
#
#        self.found = found
#
#    def found_point(self):
#        """ check if the tank has discovered a new attractive point in a tick"""
#        if self.found == True:
#            return True
#        else:
#            return False
#
#    def cap_max_and_min(self, value):
#        """ set the value within the world size"""
#        if value > 400:
#            value = 400
#        if value < -400:
#            value = -400
#        return value
#
#    def add_new_attractive_point(self):
#        """ if the tank seems to be stuck, throw in a new attractive point to get it to move again"""
#        found = False
#        for k in xrange(len(self.grids)): # grid chunk divided by # of tanks
#            for j in xrange(len(self.grids[k])): # rows
#                for i in xrange(len(self.grids[k][j])): # cols
#                    if self.flags[k][j][i].poss_color != self.constants['team']:
#                        new_x = self.cap_max_and_min(random.randint(self.flags[k][j][i].x-(self.occgrid_width/2+1), self.flags[k][j][i].x+(self.occgrid_width/2+1)))
#                        new_y = self.cap_max_and_min(random.randint(self.flags[k][j][i].y-(self.occgrid_width/2+1), self.flags[k][j][i].y+(self.occgrid_width/2+1)))           
#                        new_point = (new_x, new_y)
#                        self.flags[k][j].insert(i, self.create_flag(new_point[0], new_point[1]))
#                        if self.debug == True:
#                            print "New point at (%d, %d) is added." % new_point
#                        #self.prev = (self.mytanks[0].x, self.mytanks[0].y)
#                        found = True
#                        break
#                if found == True:
#                    break
#            if found == True:
#                break
#
#    def swap_flag_list(self):
#        """ put the current front flags at the last of the list """
#        found = False
#        for k in xrange(len(self.grids)): # grid chunk divided by # of tanks
#            for j in xrange(len(self.grids[k])):
#                for i in xrange(len(self.grids[k][j])):
#                    if self.flags[k][j][i].poss_color != self.constants['team']:
#                        lflags = self.flags[k].pop(j)
#                        self.flags[k].append(lflags)
#                        if self.debug == True:
#                            print "Send flag list (%d, %d, %d) to the last" % (i, j, k) 
#                        found = True
#                        break
#                if found == True:
#                    break
#            if found == True:
#                break
#
#    def get_direction(self, tank):
#        """ Get the moving direction based on the combined vector """ 
#        delta_x, delta_y = self.compute_attractive_vectors(tank) # compute the strongest attractive vector and the target flag
#        command = self.create_move_forward_command(tank, delta_x, delta_y)
#        self.commands.append(command)
#
#    def compute_attractive_x_and_y(self, flag, d, tank, r):
#        if d == 0:
#            d = math.sqrt((flag.x - tank.x)**2 + (flag.y-tank.y)**2)
#        else:
#            d = math.sqrt(d)
#        
#        if flag != None and flag.poss_color != self.constants['team']:
#            theta = math.atan2(flag.y-tank.y, flag.x-tank.x)
#        else:
#            theta = 0
#        if d < r:
#            delta_x = delta_y = 0
#        else:
#            cos = math.cos(theta)
#            sin = math.sin(theta)
#            if r <= d and d <= self.attractive_s + r:
#                const = self.attractive_alpha * (d - r)
#                delta_x = const * cos
#                delta_y = const * (d - r) * sin
#            elif d > self.attractive_s + r:
#                const = self.attractive_alpha * self.attractive_s
#                delta_x = const * cos
#                delta_y = const * sin        
#        return delta_x, delta_y
#
#    def count_nonvisited_flags(self):
#        """ count the number of flags that haven't been visited"""
#        counter = 0
#        for flaglist in self.flags:
#            for flags in flaglist:
#                for flag in flags:
#                    if flag.poss_color != self.constants['team']:
#                        counter += 1
#        return counter
#
#    def compute_attractive_vectors(self, tank):
#        """ compute the strongest attractive vector and return the direction and the angle """        
#        #min_d = self.infinity
#        #best_flag = None
#
#        #print "# flags: %d" % self.count_nonvisited_flags()
##
##        found = False
##
##        for flags in self.flags:
##            for flag in flags:
##                #print "flag: (%d, %d)" % (flag.x, flag.y)
##                if flag.poss_color != self.constants['team']:
##                    d = ((flag.x - tank.x)**2 + (flag.y - tank.y)**2) # get distance between tank and flag
##                    if d < min_d:
##                        min_d = d
##                        best_flag = flag
##                        found = True
##                        break
##            if found == True:
##                break
#
#        min_d, best_flag = self.find_best_flag(tank)
#
#        if self.debug == True:
#            print "next flag position: (%d, %d)" % (best_flag.x, best_flag.y)
#            print "tank position: (%d, %d)" % (tank.x, tank.y)
#        
#        #tank.goalx = flag.x    
#        #tank.goaly = flag.y
#        
#        
#        delta_x, delta_y = self.compute_attractive_x_and_y(best_flag, min_d, tank, 0)
#
#        return delta_x, delta_y
#
#    def loop_over_flaglist(self, tank):
#        """ loop over flag list and returns min_d, best_flag, and found/not found."""
#        min_d = self.infinity
#        best_flag = None
#        separated_flag_list = self.flags[tank.index]
#        found = False
#        
#        for flags in separated_flag_list:
#            for flag in flags:
#                #print "flag: (%d, %d)" % (flag.x, flag.y)
#                if flag.poss_color != self.constants['team']:
#                    d = ((flag.x - tank.x)**2 + (flag.y - tank.y)**2) # get distance between tank and flag
#                    if d < min_d:
#                        min_d = d
#                        best_flag = flag
#                        found = True
#                        break
#            if found == True:
#                break
#
#        return found, min_d, best_flag
#
#    def find_best_flag(self, tank):
#        """ find best flag and its mininum distance"""
#        found, min_d, best_flag = self.loop_over_flaglist(tank)
#        
#        if found == False:
#            for mytank in self.mytanks:
#                if mytank.index != tank.index:
#                    found, min_d, best_flag = self.loop_over_flaglist(mytank)
#                    if found == True:
#                        return min_d, best_flag
#        
#        return min_d, best_flag
#        
#    def create_move_forward_command(self, tank, delta_x, delta_y):
#        """ produce move forward command """
#        angle = self.compute_angle(tank, delta_x, delta_y)
#        self.stuck = self.is_stuck(tank)
#        if self.debug == True:
#            print "Prev: %d, %d  Cur: %d, %d  Stuck? %s" % (self.prev[tank.index][0], self.prev[tank.index][1], tank.x, tank.y, self.stuck)
#        if self.stuck == False:
#            command = Command(tank.index, self.speed, 2*angle, self.fire)
#        else:
#            #command = Command(tank.index, self.speed, 2*angle, self.fire)
#            #angle = angle + random.uniform(-5, 0)
#            command = Command(tank.index, self.speed, angle-5, self.fire)                    
#        return command
#
#    def compute_angle(self, tank, delta_x, delta_y):
#        angle = math.atan2(delta_y, delta_x)
#        relative_angle = self.normalize_angle(angle - tank.angle, tank)
#        return relative_angle
#            
#    def normalize_angle(self, angle, tank):
#        """Make any angle be between +/- pi."""
#        angle -= 2 * math.pi * int (angle / (2 * math.pi))
#        if angle <= -math.pi:
#            angle += 2 * math.pi
#        elif angle > math.pi:
#            angle -= 2 * math.pi                    
#        return angle


def main():
    # Process CLI arguments.
    try:
        execname, host, port = sys.argv
    except ValueError:
        execname = sys.argv[0]
        print >>sys.stderr, '%s: incorrect number of arguments' % execname
        print >>sys.stderr, 'usage: %s hostname port' % sys.argv[0]
        sys.exit(-1)

    # Connect.
    #bzrc = BZRC(host, int(port), debug=True)
    bzrc = BZRC(host, int(port))

    agent = Agent(bzrc)

    start = time.time()

    prev = time.time()

    time_diff = 0

    # Run the agent
    try:
        while True: 
            agent.tick(time_diff)
            #cur = time.time()
            #print "Elapsed Time: %d" % (time.time() - start)
#            if agent.found_point():
#                prev = cur
#            else:
#                if cur - prev > agent.time_threshold:
#                    if agent.count_nonvisited_flags() > 1:
#                        agent.swap_flags()
#                    else:
#                        agent.add_new_attractive_point()
#                    prev = cur
                    
    except KeyboardInterrupt:
        print "Exiting due to keyboard interrupt."
        bzrc.close()

if __name__ == '__main__':
    main()
