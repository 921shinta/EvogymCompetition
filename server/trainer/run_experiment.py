from __future__ import annotations
import os, sys, shutil, argparse, math, random
from typing import List
import numpy as np
from evogym import sample_robot, hashable, get_full_connectivity

from server.trainer.utils.mp_group import Group
from server.trainer.ga.base import Individual
from server.trainer.ga.engine import resolve_env, copy_active_assets
from server.trainer.ga.evaluator import evaluate_structure
from server.trainer.ga.registry import get_mutation, get_crossover, get_selection
from server.trainer.ga.operators.crossovers import SinglePointCrossover, NoCrossover
from server.trainer.ga.operators.mutations import RotationMutation, DefaultMutation


def save_generation(home_path: str, generation: int, structures: List[Individual]) -> None:
    gen_dir = os.path.join(home_path, f"generation_{generation}")
    struct_dir = os.path.join(gen_dir, "structure")
    os.makedirs(struct_dir, exist_ok=True)
    with open(os.path.join(gen_dir, "output.txt"), "w") as fout:
        for s in structures:
            conn = s.connections
            if conn is None:
                conn = get_full_connectivity(s.body)

            np.savez(
                os.path.join(struct_dir, f"{s.label}.npz"),
                s.body,
                conn,
                np.array(s.controller_params, dtype=np.float32),
            )
            f_str = ",".join(f"{v:.4f}" for v in s.controller_params)
            fout.write(f"{s.label}\t{s.fitness:.4f}\t{f_str}\n")


def run_experiment(
    exp_name: str,
    env_name: str | None,
    pop_size: int,
    structure_shape: tuple[int, int],
    max_evaluations: int,
    num_cores: int,
    max_steps: int,
    max_episode_steps: int | None = None,
    mutation_name: str = "default",
    mutation_rate: float = 0.1,
    crossover_name: str = "none",
    crossover_rate: float = 0.5,
    selection_name: str = "truncation",
    use_custom_env: bool = True,
) -> None:
    env_id, is_custom = resolve_env(env_name, max_episode_steps, force_custom=use_custom_env)
    home_path = os.path.join("server/saved_data", exp_name)
    if os.path.exists(home_path):
        shutil.rmtree(home_path)
    os.makedirs(home_path, exist_ok=True)
    if is_custom:
        copy_active_assets(home_path, env_id)

    metadata_path = os.path.join(home_path, "metadata.txt")
    with open(metadata_path, "w") as f:
        f.write("ALGO: GA\n")
        f.write(f"ENV: {env_id}\n")
        f.write(f"POP_SIZE: {pop_size}\n")
        f.write(f"STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n")
        f.write(f"MAX_EVALUATIONS: {max_evaluations}\n")
        f.write(f"MAX_STEPS: {max_steps}\n")
        if max_episode_steps is not None:
            f.write(f"MAX_EPISODE_STEPS: {max_episode_steps}\n")
        try:
            import evogym, gymnasium, numpy as _np
            f.write(f"VERSIONS: evogym={getattr(evogym, '__version__', 'unknown')}\n")
        except:
            pass

    if mutation_name == "default" or mutation_name == "rotation":
        base_mutation_func = DefaultMutation()
    else:
        base_mutation_func = get_mutation(mutation_name)
    
    rotation_op = RotationMutation()
    use_rotation = (mutation_name == "rotation")

    if crossover_name == "single_point":
        crossover = SinglePointCrossover()
    elif crossover_name == "none":
        crossover = NoCrossover()
    else:
        crossover = get_crossover(crossover_name)
        
    selection = get_selection(selection_name)

    structures: List[Individual] = []
    seen_hashes = set()
    num_evals, gen = 0, 0
    for i in range(pop_size):
        body, connections = sample_robot(structure_shape)
        while hashable(body) in seen_hashes:
            body, connections = sample_robot(structure_shape)
        structures.append(Individual(body, connections, i))
        seen_hashes.add(hashable(body))
        num_evals += 1

    while num_evals <= max_evaluations:
        print(f"Generation {gen} | evals {num_evals}/{max_evaluations}")
        group = Group()
        for s in structures:
            group.add_job(
                evaluate_structure,
                (s.body, s.connections, s.controller_params, env_id, max_steps),
                callback=s.set_reward,
            )
        group.run_jobs(num_cores)

        structures.sort(key=lambda x: x.fitness, reverse=True)
        save_generation(home_path, gen, structures)

        if num_evals >= max_evaluations:
            break

        survivors, lam = selection(structures, pop_size, num_evals, max_evaluations)
        children: List[Individual] = []
        next_label = len(survivors)

        while len(children) < lam and num_evals < max_evaluations:
            # 1. 交叉 (Crossover)
            if random.random() < crossover_rate:
                p1, p2 = random.sample(survivors, 2)
                c1, c2 = crossover(p1, p2)
                for child in (c1, c2):
                    if len(children) >= lam or num_evals >= max_evaluations: break
                    
                    # 【追加修正】ここで重複チェックを行う！
                    # 親のコピー（クローン）だったり、既出の形状ならスキップする
                    if hashable(child.body) in seen_hashes:
                        continue

                    child.label = next_label
                    children.append(child)
                    seen_hashes.add(hashable(child.body))
                    next_label += 1
                    num_evals += 1
            
            # 2. 突然変異 (Mutation)
            else:
                parent = random.choice(survivors)
                child = None
                
                if use_rotation and random.random() < mutation_rate:
                    child = rotation_op(parent, next_label)
                else:
                    child = base_mutation_func(parent, next_label)
                
                if child is None: continue
                # 突然変異はもともと重複チェックがある
                if hashable(child.body) in seen_hashes: continue
                
                children.append(child)
                seen_hashes.add(hashable(child.body))
                next_label += 1
                num_evals += 1

        structures = survivors + children
        gen += 1

    print("GA complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GA for EvoGym")
    parser.add_argument("--exp_name", type=str, default="default_experiment")
    parser.add_argument("--env_name", type=str, default="Walker-v0")
    parser.add_argument("--pop_size", type=int, default=120)
    parser.add_argument("--structure_shape", type=int, nargs=2, default=[5, 5])
    parser.add_argument("--max_evaluations", type=int, default=600)
    parser.add_argument("--num_cores", type=int, default=12)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--max_episode_steps", type=int, default=None)
    parser.add_argument("--mutation", type=str, default="default", help="Choices: default, rotation")
    parser.add_argument("--mutation_rate", type=float, default=0.1, help="Probability of rotation")
    parser.add_argument("--crossover", type=str, default="none", help="Choices: none, single_point")
    parser.add_argument("--crossover_rate", type=float, default=0.5)
    parser.add_argument("--selection", type=str, default="truncation")
    parser.add_argument("--custom_env", action=argparse.BooleanOptionalAction, default=True)
    
    args = parser.parse_args()

    run_experiment(
        exp_name=args.exp_name,
        env_name=args.env_name,
        pop_size=args.pop_size,
        structure_shape=tuple(args.structure_shape),
        max_evaluations=args.max_evaluations,
        num_cores=args.num_cores,
        max_steps=args.max_steps,
        max_episode_steps=args.max_episode_steps,
        mutation_name=args.mutation,
        mutation_rate=args.mutation_rate,
        crossover_name=args.crossover,
        crossover_rate=args.crossover_rate,
        selection_name=args.selection,
        use_custom_env=args.custom_env,
    )