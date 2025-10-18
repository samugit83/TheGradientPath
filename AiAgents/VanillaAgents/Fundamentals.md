# Building Tiny Agents from Scratch: Python Class Fundamentals

Welcome to your journey into building AI agents with pure Python! In this tutorial, we'll explore the essential class features that form the backbone of any agent system. Each concept builds toward creating intelligent, autonomous programs that can think, remember, and act.

---

## 1. Minimal Class & Instance Attributes

**Voiceover Script:**
Think of an agent as a digital assistant that needs to remember things about itself - like its name, current task, or location in a virtual world. But here's the magic: when we create multiple agents, each one needs its own separate memory bank. That's exactly what Python classes and instance attributes give us.

When we define a class, we're creating a blueprint - like an architectural plan for a house. The `__init__` method is like the construction process that happens when we build each individual house from that blueprint. Every time we call `SimpleAgent("Alice", "analyzing data")`, Python creates a brand new agent object with its own personal storage space.

The `self` parameter is crucial here - it's Python's way of saying "this specific agent instance." When we write `self.name = agent_name`, we're storing the name in this particular agent's memory, not affecting any other agents. This separation is fundamental to building agent systems where each agent can have different states, tasks, and behaviors while sharing the same underlying structure. Think of it like having a factory that produces unique robots - same design, but each robot has its own identity and mission.

```python
class SimpleAgent:
    """A basic agent that remembers its identity and current task."""
    
    def __init__(self, agent_name, initial_task):
        self.name = agent_name
        self.current_task = initial_task
        self.energy = 100  # All agents start with full energy
    
    def introduce(self):
        return f"Hi! I'm {self.name}, currently working on: {self.current_task}"

# Create two different agents
alice = SimpleAgent("Alice", "analyzing data")
bob = SimpleAgent("Bob", "monitoring systems")

print(alice.introduce())  # Hi! I'm Alice, currently working on: analyzing data
print(bob.introduce())    # Hi! I'm Bob, currently working on: monitoring systems
print(f"Alice's energy: {alice.energy}")  # Alice's energy: 100
```

**Takeaway:** Classes create blueprints for agents, and each agent instance gets its own personal memory storage.

---

## 2. Default & Keyword Args in `__init__`

**Voiceover Script:**
Real agents need flexibility when they're created, but we also want to make agent creation as simple as possible for common scenarios. This is where Python's default and keyword arguments become incredibly powerful for agent design.

Default arguments are like having intelligent defaults in a configuration system. When we write `role="assistant"`, we're saying "unless someone specifically tells me otherwise, assume this agent is an assistant." This makes creating standard agents super easy - just provide a name and you're done. But if you need something special, you can override that default.

Now here's where it gets really interesting: the asterisk `*` in our parameter list creates a boundary. Everything after it must be specified by name, not position. This prevents dangerous mistakes like accidentally swapping the `active` and `shift` parameters. Imagine deploying a night-shift security agent that you accidentally marked as inactive, or vice versa! Keyword-only arguments force explicit, clear configuration.

In our example, notice how `FlexibleAgent("Charlie")` uses all defaults, but `FlexibleAgent("Dana", "monitor", active=True, shift="night")` shows explicit configuration. The `active=True, shift="night"` part is crystal clear - there's no ambiguity about what we're setting. This design pattern becomes crucial when managing large agent fleets where configuration errors could have serious consequences.

```python
class FlexibleAgent:
    """An agent with sensible defaults and explicit configuration options."""
    
    def __init__(self, name, role="assistant", *, active=True, shift="day"):
        self.name = name
        self.role = role
        self.active = active
        self.shift = shift
        self.tasks_completed = 0
    
    def status_report(self):
        status = "active" if self.active else "standby"
        return f"{self.name} ({self.role}) - {status} on {self.shift} shift"
    
    def complete_task(self):
        if self.active:
            self.tasks_completed += 1
            return f"{self.name} completed task #{self.tasks_completed}"
        return f"{self.name} is in standby mode"

# Different ways to create agents
standard_agent = FlexibleAgent("Charlie")  # Uses all defaults
night_agent = FlexibleAgent("Dana", "monitor", active=True, shift="night")
standby_agent = FlexibleAgent("Echo", active=False)

print(standard_agent.status_report())  # Charlie (assistant) - active on day shift
print(night_agent.complete_task())     # Dana completed task #1
print(standby_agent.complete_task())   # Echo is in standby mode
```

**Takeaway:** Default arguments create user-friendly agent constructors while keyword-only arguments prevent configuration mistakes.

---

## 3. Class Attributes vs Instance Attributes

**Voiceover Script:**
Imagine you're managing a team of agents in a control room. Some information belongs to individual agents - like their personal task queue, energy level, or current location. But other information needs to be shared across the entire team - like the total number of agents deployed, the team name, or system-wide configuration settings.

This is where the distinction between class and instance attributes becomes critical for agent architecture. Class attributes are defined directly in the class body, outside any methods. They're like a shared bulletin board that all agents can read and modify. When we write `total_agents = 0` at the class level, we're creating a single counter that's shared by every agent instance.

But here's what's fascinating: when an agent updates a class attribute like `Counter.total_agents += 1`, that change is immediately visible to all other agents. It's truly shared memory! This is perfect for tracking system-wide statistics, configuration settings, or coordination information that the entire agent team needs to know about.

Instance attributes, on the other hand, are personal to each agent. When we write `self.energy = self.max_energy`, each agent gets its own separate energy value. Agent A001 might have 90% energy while Agent A002 has full energy - they're completely independent.

The `@classmethod` decorator is the cherry on top - it lets us create methods that work with the class itself rather than individual instances. `team_status()` can be called even if no agents exist yet, because it operates on the shared class-level data. This pattern is incredibly useful for agent management and monitoring systems.

```python
class TeamAgent:
    """An agent that tracks both personal and team-wide information."""
    
    # Class attributes - shared by all agents
    total_agents = 0
    team_name = "Alpha Squad"
    max_energy = 100
    
    def __init__(self, agent_id):
        # Instance attributes - unique to each agent
        self.id = agent_id
        self.energy = self.max_energy  # Start with full energy
        self.personal_tasks = []
        
        # Update the shared counter
        TeamAgent.total_agents += 1
    
    def take_task(self, task):
        self.personal_tasks.append(task)
        self.energy -= 10  # Tasks consume energy
        return f"Agent {self.id} took task: {task}"
    
    @classmethod
    def team_status(cls):
        return f"Team '{cls.team_name}' has {cls.total_agents} active agents"

# Create a team of agents
agent1 = TeamAgent("A001")
agent2 = TeamAgent("A002")
agent3 = TeamAgent("A003")

print(TeamAgent.team_status())  # Team 'Alpha Squad' has 3 active agents
print(agent1.take_task("scan network"))  # Agent A001 took task: scan network
print(f"Agent A001 energy: {agent1.energy}")  # Agent A001 energy: 90
print(f"Total team size: {TeamAgent.total_agents}")  # Total team size: 3
```

**Takeaway:** Use class attributes for shared team data and instance attributes for personal agent data.

---

## 4. `__slots__` to Save Memory / Prevent Typos

**Voiceover Script:**
When you're running dozens or hundreds of agents, memory efficiency becomes critical - especially if you're deploying on edge devices or managing large agent swarms. But there's another subtle problem that can plague agent systems: attribute typos that create silent bugs.

Here's what happens normally: every Python object carries around a dictionary called `__dict__` that stores all its attributes. This dictionary is flexible - you can add new attributes anytime - but it also consumes extra memory for the dictionary overhead. When you have thousands of agents, this adds up quickly.

`__slots__` changes the game completely. When we define `__slots__ = ("agent_id", "status", "current_location", "battery_level")`, we're telling Python: "This object will only ever have these four attributes, period." Python then creates a much more efficient internal structure - no dictionary overhead, just direct memory slots for each attribute.

But here's the bonus feature that makes `__slots__` invaluable for agent systems: typo prevention. Without `__slots__`, if you accidentally write `scout.battry_level = 50` instead of `scout.battery_level = 50`, Python happily creates a new attribute with the misspelled name. Your agent might keep operating, but now it's reading from the wrong variable! This kind of bug can hide for months in production systems.

With `__slots__`, that typo would immediately raise an AttributeError, failing fast and forcing you to fix the bug. In agent systems where reliability is crucial - like autonomous vehicles or trading bots - this immediate failure detection can save you from catastrophic errors. The memory savings are nice, but the typo protection might be even more valuable.

```python
class EfficientAgent:
    """A memory-efficient agent that prevents attribute typos."""
    
    # Only these attributes are allowed - saves memory and prevents typos
    __slots__ = ("agent_id", "status", "current_location", "battery_level")
    
    def __init__(self, agent_id, starting_location="base"):
        self.agent_id = agent_id
        self.status = "idle"
        self.current_location = starting_location
        self.battery_level = 100.0
    
    def move_to(self, new_location):
        if self.battery_level < 10:
            return f"Agent {self.agent_id}: Low battery! Cannot move."
        
        self.current_location = new_location
        self.battery_level -= 5.0
        self.status = "moving"
        return f"Agent {self.agent_id} moving to {new_location}"
    
    def report_position(self):
        return f"Agent {self.agent_id} at {self.current_location} (battery: {self.battery_level}%)"

# Create efficient agents
scout = EfficientAgent("Scout-1", "perimeter")
guard = EfficientAgent("Guard-2")

print(guard.move_to("checkpoint-alpha"))  # Agent Guard-2 moving to checkpoint-alpha
print(scout.report_position())  # Agent Scout-1 at perimeter (battery: 100.0%)

# This would raise an AttributeError - typo protection in action!
# scout.battry_level = 50  # ❌ Would fail - 'battry_level' is not in __slots__
```

**Takeaway:** `__slots__` makes agents memory-efficient and typo-proof - perfect for large agent deployments.

---

## 5. `dataclasses` for Boilerplate-Free Models

**Voiceover Script:**
Sometimes you need to create agent data structures quickly without writing tons of repetitive code. Python's dataclasses are like having a smart assistant that writes all the boring setup code for you, and they're particularly powerful for agent systems where you need clean, well-structured data models.

Think about what we usually have to write for a typical class: an `__init__` method to set up attributes, a `__repr__` method for debugging output, `__eq__` and other comparison methods if we want to compare objects, maybe `__hash__` if we want to use objects as dictionary keys. That's a lot of boilerplate code that's the same pattern over and over.

The `@dataclass` decorator eliminates all that repetitive work. When we write `@dataclass` above our class definition, Python automatically generates the `__init__`, `__repr__`, and comparison methods based on the type hints we provide. But here's what makes dataclasses perfect for agents: the `field()` function gives us sophisticated control over attribute initialization.

Look at `assigned_agents: List[str] = field(default_factory=list)` in our Mission class. We can't just write `assigned_agents: List[str] = []` because that would create a single list shared between all Mission instances - a classic Python gotcha! The `field(default_factory=list)` tells Python to create a fresh, empty list for each new Mission instance.

The real power shows when we create objects. `Mission("M001", 9, "Secure the data center")` automatically gets all its attributes set up, and when we print it, we get a beautiful, readable representation showing all the values. No manual `__repr__` method needed! This makes debugging agent systems much easier because you can instantly see the state of any mission or agent object.

For agent systems dealing with complex data structures - missions, tasks, communication messages, sensor readings - dataclasses eliminate tedious code while providing robust, debuggable objects with automatic features that make development much more pleasant.

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class Mission:
    """A mission that agents can be assigned to complete."""
    mission_id: str
    priority: int
    description: str
    status: str = "pending"
    assigned_agents: List[str] = field(default_factory=list)
    
    def assign_agent(self, agent_id: str):
        if agent_id not in self.assigned_agents:
            self.assigned_agents.append(agent_id)
            return f"Agent {agent_id} assigned to mission {self.mission_id}"
        return f"Agent {agent_id} already assigned to this mission"
    
    def is_high_priority(self) -> bool:
        return self.priority >= 8

@dataclass
class SmartAgent:
    """A modern agent using dataclass for clean, automatic features."""
    name: str
    specialization: str
    experience_level: int = 1
    active_missions: List[Mission] = field(default_factory=list)
    
    def take_mission(self, mission: Mission):
        self.active_missions.append(mission)
        mission.assign_agent(self.name)
        return f"{self.name} accepted mission: {mission.description}"

# Create missions and agents with minimal code
urgent_mission = Mission("M001", 9, "Secure the data center")
routine_mission = Mission("M002", 5, "System health check")

expert_agent = SmartAgent("Alex", "security", experience_level=5)
junior_agent = SmartAgent("Jordan", "maintenance")

print(expert_agent.take_mission(urgent_mission))
# Alex accepted mission: Secure the data center

print(f"Mission details: {urgent_mission}")
# Mission details: Mission(mission_id='M001', priority=9, description='Secure the data center', status='pending', assigned_agents=['Alex'])

print(f"Is urgent mission high priority? {urgent_mission.is_high_priority()}")  # True
print(f"Agent info: {expert_agent}")
# Agent info: SmartAgent(name='Alex', specialization='security', experience_level=5, active_missions=[Mission(...)])
```

**Takeaway:** Dataclasses eliminate boilerplate code and provide automatic features like string representation and comparison - perfect for rapid agent prototyping.

---

## What's Next?

Congratulations! You now have the fundamental building blocks for creating agent systems. These five concepts - basic classes, flexible initialization, shared vs personal data, memory efficiency, and boilerplate-free models - form the foundation of every agent architecture.

In the next tutorial, we'll combine these fundamentals to build agent communication systems, task queues, and simple decision-making workflows. Your agents are about to get much smarter!

---

*Happy coding! 🤖*
