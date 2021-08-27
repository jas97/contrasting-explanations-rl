from or_gym.envs.classic_or import VehicleRoutingEnv


class VehicleRouting(VehicleRoutingEnv):

    def     __init__(self, grid, order_miss_penalty, *args, **kwargs):
        super().__init__(args, kwargs)

        # self.grid = grid
        self.order_miss_penalty = order_miss_penalty

        self.scaler = None

    def render(self):
        rest_pos_0, rest_pos_1 = self.state[0:2], self.state[2:4]  # 0:4
        driver_loc = self.state[4:6]
        vehicle_load = self.state[6]
        vehicle_cap = self.state[7]

        orders = []
        start = 8
        end = 14
        for i in range(self.max_orders):
            orders += [self.state[start: end]]
            start = end
            end = start + 6

        s = ''
        for j in range(self.grid[0]):
            row = ''
            for i in range(self.grid[0]):
                if i == driver_loc[0] and j == driver_loc[1]:
                    row += ' D '
                elif i == rest_pos_0[0] and j == rest_pos_0[1]:
                    row += ' A '
                elif i == rest_pos_1[0] and j == rest_pos_1[1]:
                    row += ' B '
                else:
                    row += ' - '

            s += row
            s += '\n'

        for i in range(self.max_orders):
            o = orders[i]
            s += 'Order {}: restaurant: {} destination: ({}, {}), status: {}, time since: {}, value: {}\n'.format(i, *o)

        s += 'Vehicle load: {} Vehicle capacity: {}\n'.format(vehicle_load, vehicle_cap)

        print(s)

    def update_state(self, state):
        self.state = state

    def set_steps_elapsed(self, steps):
        self.step_count = steps

    def get_steps_elapsed(self):
        return self.step_count