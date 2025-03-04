import numpy as np
import torch


class CurriculumScheduler:
    """Curriculum learning scheduler for molecular structure generation."""

    def __init__(self, num_stages=5, mutations_per_stage=(1, 2, 3, 5, 10),
                 stage_episodes=(100, 200, 300, 400, 500), min_success_rate=0.7):
        """
        Initialize curriculum scheduler.

        Args:
            num_stages: Number of curriculum stages
            mutations_per_stage: Mutations allowed in each stage
            stage_episodes: Episodes per stage
            min_success_rate: Minimum success rate to advance stage
        """
        self.num_stages = num_stages
        self.mutations_per_stage = mutations_per_stage
        self.stage_episodes = stage_episodes
        self.min_success_rate = min_success_rate

        self.current_stage = 0
        self.episode_in_stage = 0
        self.success_history = []

    def get_current_task_parameters(self):
        """
        Get parameters for current curriculum stage.

        Returns:
            dict: Parameters for current stage
        """
        # Get mutations allowed in current stage
        max_mutations = self.mutations_per_stage[self.current_stage]

        # More complex tasks in later stages
        if self.current_stage <= 1:
            # Early stages: only conservative point mutations
            mutation_type = "conservative"
            target_region = "random"
        elif self.current_stage <= 3:
            # Middle stages: allow non-conservative mutations in non-critical regions
            mutation_type = "any" if np.random.random() > 0.3 else "conservative"
            target_region = "non_critical" if np.random.random() > 0.3 else "random"
        else:
            # Advanced stages: allow any mutation anywhere, including loops
            mutation_type = "any"
            target_region = np.random.choice(["random", "loop", "non_critical", "critical"],
                                             p=[0.2, 0.3, 0.3, 0.2])

        return {
            'max_mutations': max_mutations,
            'mutation_type': mutation_type,
            'target_region': target_region,
            'stage': self.current_stage,
            'stage_progress': self.episode_in_stage / self.stage_episodes[self.current_stage]
        }

    def update(self, success):
        """
        Update curriculum based on episode result.

        Args:
            success: Whether the episode was successful

        Returns:
            bool: Whether the stage advanced
        """
        self.success_history.append(float(success))
        self.episode_in_stage += 1

        # Keep only recent history
        if len(self.success_history) > 100:
            self.success_history = self.success_history[-100:]

        # Check if we should advance to the next stage
        stage_advanced = False
        if self.episode_in_stage >= self.stage_episodes[self.current_stage]:
            # Calculate success rate
            success_rate = np.mean(self.success_history[-self.episode_in_stage:])

            # Advance if success rate is high enough
            if success_rate >= self.min_success_rate and self.current_stage < self.num_stages - 1:
                self.current_stage += 1
                self.episode_in_stage = 0
                stage_advanced = True
                print(f"Curriculum advanced to stage {self.current_stage}")
            else:
                # Reset episode counter but stay in same stage
                self.episode_in_stage = 0

        return stage_advanced

    def get_stage_info(self):
        """
        Get information about current curriculum stage.

        Returns:
            dict: Stage information
        """
        if len(self.success_history) > 0:
            recent_success_rate = np.mean(self.success_history[-min(len(self.success_history),
                                                                    self.episode_in_stage):])
        else:
            recent_success_rate = 0.0

        return {
            'stage': self.current_stage,
            'episode_in_stage': self.episode_in_stage,
            'total_episodes': sum(self.stage_episodes[:self.current_stage]) + self.episode_in_stage,
            'max_mutations': self.mutations_per_stage[self.current_stage],
            'success_rate': recent_success_rate,
            'episodes_until_evaluation': self.stage_episodes[self.current_stage] - self.episode_in_stage
        }
