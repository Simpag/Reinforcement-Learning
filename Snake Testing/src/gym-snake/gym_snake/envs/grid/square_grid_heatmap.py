from gym_snake.envs.constants import Action4, Direction4
from gym_snake.envs.grid.base_grid import BaseGrid


class SquareGrid(BaseGrid):

    def __init__(self, *args, **kwargs):
        super(SquareGrid, self).__init__(*args, **kwargs)

    def move(self, actions):
        assert not self.all_done

        rewards = [self.reward_none] * self.num_snakes
        num_new_apples = 0

        # Move live snakes and eat apples
        if not self.always_expand:
            for snake, action in zip(self.snakes, actions):
                if snake.alive:
                    # Only contract if not about to eat apple
                    next_head = snake.next_head(action)
                    if next_head not in self.apples:
                        snake.contract()

        for i, snake, action in zip(range(self.num_snakes), self.snakes, actions):
            if not snake.alive:
                continue

            next_head = snake.next_head(action)
            if self.is_blocked(next_head):
                snake.kill()
                rewards[i] = self.reward_collision
            else:
                snake.expand(action)
                if next_head in self.apples:
                    if self.done_apple:
                        snake.kill()
                    self.apples.remove(next_head)
                    num_new_apples += 1
                    rewards[i] = self.reward_apple
                else:
                    rewards[i] = self._get_apple_closeness()

        # If all agents are done, mark grid as done (and prevent future moves)
        dones = [not snake.alive for snake in self.snakes]
        self.all_done = False not in dones

        # Create new apples
        self.add_apples(num_new_apples)

        return rewards, dones

    def _get_apple_closeness(self, distance):
        # Returns a reward if the snake head is within 1, 2, 3 moves
        # Only works for one apple and one snake
        # Max distance is 512
        head = self.snakes[0].get_head_pos() # (x, y)
        apple = self.apples.get_pos()        # (x, y)

        d = (head[0] - apple[0])**2 + (head[1] - apple[1])**2 # Square distance to skip on square root calculation
        
        if d == 1: # one move away
            return max(self.reward_apple//2, 0)
        elif d < 5: # 2 moves away
            return max(self.reward_apple//4, 0)
        elif d < 10: # 3 moves away
            return 1

        return 0


    def get_forward_action(self):
        return Action4.forward

    def get_random_direction(self):
        return Direction4(self.np_random.integers(0, len(Direction4)))

    def get_renderer_dimensions(self, tile_size):
        return self.width * tile_size, self.height * tile_size

    def render(self, r, tile_size, cell_pixels):
        r_width, r_height = self.get_renderer_dimensions(tile_size)
        assert r.width == r_width
        assert r.height == r_height

        # Total grid size at native scale
        width_px = self.width * cell_pixels
        height_px = self.height * cell_pixels

        r.push()

        # Internally, we draw at the "large" full-grid resolution, but we
        # use the renderer to scale back to the desired size
        r.scale(tile_size / cell_pixels, tile_size / cell_pixels)

        # Draw the background of the in-world cells black
        r.fillRect(
            0,
            0,
            width_px,
            height_px,
            0, 0, 0
        )

        # Draw grid lines
        r.setLineColor(100, 100, 100)
        r.setLineWidth(cell_pixels / tile_size)
        for rowIdx in range(0, self.height):
            y = cell_pixels * rowIdx
            r.drawLine(0, y, width_px, y)
        for colIdx in range(0, self.width):
            x = cell_pixels * colIdx
            r.drawLine(x, 0, x, height_px)

        # Render the objects
        snake_cell_renderer = SquareGrid._square_cell_renderer(r, cell_pixels)
        for snake in self.snakes:
            snake.render(snake_cell_renderer)

        apple_cell_renderer = SquareGrid._circle_cell_renderer(r, cell_pixels)
        self.apples.render(apple_cell_renderer)

        r.pop()

    @staticmethod
    def _square_cell_renderer(r, cell_pixels):
        points = (
            (0,           cell_pixels),
            (cell_pixels, cell_pixels),
            (cell_pixels, 0),
            (0,           0)
        )

        def cell_renderer(p, color):
            x, y = p

            r.push()
            r.setLineColor(*color)
            r.setColor(*color)
            r.translate(x * cell_pixels, y * cell_pixels)
            r.drawPolygon(points)
            r.pop()

        return cell_renderer

    @staticmethod
    def _circle_cell_renderer(r, cell_pixels):
        center_coordinate = cell_pixels / 2
        circle_r = cell_pixels * 10 / 32

        def cell_renderer(p, color):
            x, y = p

            r.push()
            r.setLineColor(*color)
            r.setColor(*color)
            r.translate(x * cell_pixels, y * cell_pixels)
            r.drawCircle(center_coordinate, center_coordinate, circle_r)
            r.pop()

        return cell_renderer
