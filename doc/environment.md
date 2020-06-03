Environment
===========

Environments are a wrapper to interact with **houses**.

House
-----
A house has a list of **elevators**, which have their own state.

### Requests
To handle the passenger requests, there are four arrays, which all have the length of
the number of floors in the house:

- ``up_requests`` 
- ``down_requests``

These are boolean arrays which store for every floor if there are requests of passengers
wanting to go up or down respectively.

- ``up_requests_waiting_since``
- ``down_requests_waiting_since``

These are float arrays which store the time of the creation of the respective request.
They are only meaningful if the respective bool in the ``up_requests`` or
``down_requests`` array are ``True``.

### Time
Each house as well as each elevator has a time. The time of the elevators should always
be larger or equal to the time of the house. To perform actions, the environment always
chooses the elevator with the lowest time, as this is the one, which needs new control
instructions first. After the elevator is controlled, the time of the house elapses to
the new minimum time of all elevators. When time is elapsed, the **PassengerGenerator**
creates new requests.

### Actions
An action consists of an elevator-id and an elevator-action.
Possible actions are _up_, _down_, and _open_.
When execution an _open_ action, the elevator first lets passengers exit and then enter.
Each action increases the time of the elevator.

Elevators
---------
Each elevator has a set of passengers as well as a boolean array for the requested floor
of its passengers.
Elevators can move up or down an they can let passengers enter or exit the car.
When an elevator lets people enter at a floor, the **PassengerGenerator** samples
new **Passengers** based on the request and the waiting time of the floor.
Elevator do also have a maximum capacity.
When more passengers want to enter than there is room for, the request does not
disappear and the waiting since time gets updated.

Passengers
----------
Passengers are created by the **PassengerGenerator** when they enter an elevator.
They have a distribution of possible target floors and make the decision of whether they
leave the car at a certain floor just when the elevator opens at that floor. This
ensures, that an agent cannot know how many passengers will leave at a certain floor
by simulating the environment.
When a passenger decides to not exit at a certain floor, their target distribution
changes so that they have a probability of zero to exit at this floor in the future.
Each passenger has a ``waiting_since`` attribute, to keep track of his waiting time.

PassengerGenerator
------------------
The purpose of the PassengerGenerator is to generate requests to the whole elevator
system as well as to generate passengers when they enter an elevator.

### Generating Requests
When a house elapses time ``t``, the PassengerGenerator samples new requests.
There is a ``rate`` for each floor, which determines the frequency of passengers
appearing at this floor and requesting a drive.
We sample the count of passengers appearing at each floor from a Poisson distribution
~Poi(``rate``).
Then we sample a target floor for each passenger according to a pre-defined Categorical
distribution over targets.
Usually the rate and target probability of the ground floor is higher than others.
Based on the target floors, we can determine for each passenger, if he wants to go up or
down.
We also sample an arrival time for each passenger from a Uniform distribution of the
time interval from ``0`` to ``t`` to which we add the current time of the house.
Having the arrival times and directions of each passenger, we can update the values of
``up_requests``, ``down_requests``, ``up_requests_waiting_since``, and
``down_requests_waiting_since``.
Note that only the passengers with the lowest arrival times can affect the request
signals.
If an older request on the same floor going in the same direction already exists, new
passengers do not have any effect.
This procedure is only relevant for generating the request signals, all sampled values,
which are not stored in the request arrays are discarded.

### Generating Entering Passengers
When an elevator opens at a floor at which a request signal is set, the
PassengerGenerator generates actual passengers, which then can enter.
We assume the number of passengers to follow a Poisson distribution, but here we need
exact arrival times for each passenger.
These are computed, by iteratively sampling the durations between two passengers
arriving from an Exponential distribution ~Exp(``1 / rate``).
The first arrival time is given by the value in ``up_requests_waiting_since`` or
``down_requests_waiting_since``.
We then add the sampled durations for each new passenger until the time would be larger
than the current time of the elevator, so in the future.

If all passengers can enter, the requests disappear from ``up_requests`` and
``down_requests``.
If there is not enough capacity for all passengers, the values in
``up_requests_waiting_since`` and ``down_requests_waiting_since`` are updated to the
arrival time of the first passenger which cannot enter the elevator anymore.

We also draw a target floor for each passenger.
We use the set of all sampled target floors ``target_set`` to generate the target
floor distributions of each passenger.
To make sure that at least one passenger leaves at every requested floor, we create
one passenger for each target floor, which has a probability of 100% to leave at this
floor.
All remaining passengers get an equal distribution over all target floors which are in
``target_set``.

Why are Request and Passengers Generated so Complicatedly?
----------------------------------------------------------
The easiest way would be to generate passengers in each time step and let them have a
fixed target floor and arrival time and just hide this information from the agent to
simulate this hidden state space.
In this project however we want to use a variant of the AlphaZero algorithm, which
uses simulation of the environment to search for good actions.
If simulations are used with this hidden state space, information like the count of
passengers waiting at a floor, or the count of passengers wanting to leave at a certain
floor could leak through the observations of future time steps.
We avoid this problem by sampling these values randomly at the time they are needed.
This way, the agent can observe their distributions but it cannot determine any exact
values.
