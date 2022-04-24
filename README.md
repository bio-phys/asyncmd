# asyncmd

[![Build Status](https://drone.kotspeicher.de/api/badges/AIMMD/asyncmd/status.svg)](https://drone.kotspeicher.de/AIMMD/asyncmd)

## Synopsis

asyncmd is a library to write **concurrent** code to run and analyze molecular dynamics simulations using pythons **async/await** synthax.

## Motivation

Molecular dynamics simulations are fun and we can learn a lot about the simulated system. Running many molecular dynamics simulations of the same system concurrently is tedious, error-prone and boring but we can learn even more about the simulated system and are more efficient in doing so.
This library addresses the tedious, error-prone and boring part of setting up many similar simulations, but it leaves you with the fun part of understanding the simulated system.

## Code Example

Run 4 gromacs engines concurently from the same starting configuration (`conf.trr`) for `10000` integration steps each:
```python
import asyncmd
import asyncmd.gromacs as asyncgmx

init_conf = asyncmd.Trajectory(trajectory_file="conf.trr", structure_file="conf.gro")
mdps = [asyncgmx.MDP("config.mdp") for _ in range(4)]
# MDConfig objects (like MDP) behave like dictionaries and are easy to modify
for i, mdp in enumerate(mdps):
    # here we just modify the output frequency for every engine separately
    # but you can set any mdp option like this
    # Note how the values are in the correct types? I.e. that they are ints?
    mdp["nstxout"] *= (i + 1)
    mdp["nstvout"] *= (i + 1)
engines = [asyncgmx.GmxEngine(mdp=mdp, gro_file="conf.gro", top_file="topol.top",
                              # optional (can be omited or None), however naturally without an index file
                              # you can not reference custom groups in the .mdp-file or MDP object
                              ndx_file="index.ndx",
                              )
           for mdp in mdps]

await asyncio.gather(*(e.prepare(starting_configuration=init_conf,
                                 workdir=".", deffnm=f"engine{i}")
                       for i, e in enumerate(engines))
                       )
                     )

trajs = await asyncio.gather(*(e.run_steps(nsteps=10000) for e in engines))
```

For an in-depth introduction see also the `examples` folder in this repository which contains jupyter notebooks on various topics.

## Installation

Provide code examples and explanations of how to get the project.

## API Reference

Depending on the size of the project, if it is small and simple enough the reference docs can be added to the README. For medium size to larger projects it is important to at least provide a link to where the API reference docs live.

## Tests

Describe and show how to run the tests with code examples.

## Developers

Let people know how they can dive into the project, include important links to things like wiki, issue trackers, coding style guide, irc, twitter accounts if applicable.

## Contributors

You could (and should) give props to all the people who contributed to the code.

## License

GPL v3 (or later)

---
<sub>This README.md is printed from 100% recycled electrons.</sub>
