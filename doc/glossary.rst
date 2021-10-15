.. _glossary:


##################
 Glossary of Terms
##################

.. attention::	This is a quick hack-page to start to see (and be able to search) for 
				synonyms we might have in usage of recurring terms.  Need to consider how to transform
				and apply rst substitutions directives, which we'd probably want to enforce
				on a project-wide basis via a general "substitutions" file that is imported
				on all documents.  An aspiration for development over time.

..	glossary::

	component
	components
	Component
		A component, or sometimes emphasized as "component class" in this documentation,
		is a grouping of :term:`devices` with similar characteristics, i.e. attribute schema,
		and behavior within the :term:`network`.  All available (default) components are listed
		in the ``component.csv`` file in the main package folder.  Each component's attribute 
		structure is defined by the a csv file, corresponding to the component :term:`list_name`
		located in the ``component_attrs`` sub-directory.

		The formal identifying label for the component class itself is the CapWords formatted
		string in the first column of the ``components.csv`` file.  However, components are not
		formal Python classes.  See :ref:`design-network` for further context and details.

	devices
	device
		A device is a reference to a specific, uniquely named data element within a :term:`component`
		class.  So, for instance, a specific generating unit, might be be assigned the ``name``
		"Blue Ridge Wind Farm II", a device that would exist within the "Generator" component class.
		See :ref:`design-network` for further context and details.

	network
	Network
	pypsa.Network
		The fundamental overall container for all PyPSA data and key methods, defined as a formal
		Python class object.  Details on the Network are provided in :ref:`design-network` and 
		further supporting documentation, e.g. the :ref:`API reference<api>`.
		
		The capitalization of references to network (or Network) are not overly critical in this
		documentation.  However, we tend to use the capitalized variant "Network" when referring
		to aspects of the class definition itself and use the lower_case variable ``network`` when 
		referring to an instantiation of a specific data object, e.g. with examples related to retrieving data.
		
	list_name
		A reference that defines the string accessor / variable that provides data for all 
		:term:`devices` within a :term:`component` class.
		It is identifiable as such by being formatted in snake_case.

	snapshots
	snapshot
		Another good one to have a definition for and use when referenced

