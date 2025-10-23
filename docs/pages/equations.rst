Physics
=======

``jaxion`` solves the following equations:


Fuzzy Dark Matter
-----------------

Fuzzy dark matter is represented by the wave function, :math:`\psi`, normalized such that its density is :math:`\rho_{\rm dm}=|\psi|^2`. The field has a boson mass of :math:`m` and is evolved according to the Schrödinger–Poisson equations:

.. math::

  i\,\frac{\partial \psi}{\partial t} = -\frac{\hbar}{2m}\,\nabla^2 \psi + \frac{m}{\hbar}\,V \psi

.. math::

  \nabla^2 V = 4\pi G \left(\rho_{\rm tot} - \overline{\rho}_{\rm tot}\right)

where the gravitational potential :math:`V` is sourced by the
total density of matter: :math:`\rho_{\rm tot}=\rho_{\rm dm}+\rho_{\rm gas}+\rho_{\rm stars}`, and :math:`\overline{\rho}_{\rm tot}` is the mean density in the periodic box.

For cosmological simulations, one can define:

.. math::

  \tilde{\mathbf{r}} = a^{-1}\mathbf{r}, \qquad d\tilde{t} = a^{-2} dt, \qquad \tilde{\psi} = a^{3/2}e^{-imHr^2/(2\hbar)}\psi, \qquad \tilde{V} = a^2 V

to rewrite the equations as:

.. math::

  i\,\frac{\partial \tilde{\psi}}{\partial \tilde{t}} = -\frac{\hbar}{2m}\,\tilde{\nabla}^2 \tilde{\psi} + \frac{m}{\hbar}\,\tilde{V} \tilde{\psi}

.. math::

  \tilde{\nabla}^2 \tilde{V} = 4\pi G a \left(\tilde{\rho}_{\rm tot} - \overline{\tilde{\rho}}_{\rm tot}\right)

where :math:`a` is the cosmological scale factor, :math:`H\equiv\dot{a}/a` is the Hubble parameter, and :math:`\tilde{t}` is super-comoving time.


Gas
---

For the gas, we consider the compressible isothermal Euler equations, with density :math:`\rho_{\rm gas}\equiv\rho`, velocity :math:`\mathbf{v}`, and (fixed) sound-speed :math:`c_s`:

.. math::

  \frac{\partial}{\partial t}
  \begin{pmatrix}
  \rho \\
  \rho \mathbf{v}
  \end{pmatrix}
  +
  \nabla \cdot
  \begin{pmatrix}
  \rho \mathbf{v} \\
  \rho \mathbf{v}\mathbf{v}^T + \rho c_s^2
  \end{pmatrix}
  =
  \begin{pmatrix}
  0 \\
  -\rho \nabla V
  \end{pmatrix}

(where the gas pressure is :math:`P=\rho c_s^2`).


Stars
-----

Finally, star particles (position :math:`\mathbf{x}_s`, velocity :math:`\mathbf{v}_s`, mass :math:`m_s`) evolve according to the collisionless Boltzmann equation:

.. math::

  \frac{d\mathbf{x}_s}{dt} = \mathbf{v}_s, \qquad \frac{d\mathbf{v}_s}{dt} = -\nabla V
