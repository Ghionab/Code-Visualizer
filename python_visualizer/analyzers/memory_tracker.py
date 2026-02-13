"""
Memory tracker module for allocation and reference counting.

This module provides comprehensive memory tracking capabilities including:
- Object allocation monitoring using tracemalloc
- Reference count tracking
- Memory layout visualization
- Object mutation tracking
"""

import sys
import tracemalloc
import time
from typing import Dict, Any, Optional, List
from python_visualizer.core.data_models import AllocationInfo, MemoryLayout


class MemoryTracker:
    """
    Tracks memory allocations, reference counts, and object mutations.
    
    Uses Python's tracemalloc module for allocation tracking and sys.getrefcount
    for reference counting. Provides detailed memory layout information for
    educational purposes.
    """
    
    def __init__(self):
        """Initialize the memory tracker."""
        self.allocations: Dict[int, AllocationInfo] = {}
        self.reference_counts: Dict[int, int] = {}
        self.tracking_enabled: bool = False
        self._snapshot_before: Optional[tracemalloc.Snapshot] = None
        
    def start_tracking(self) -> None:
        """
        Initialize tracemalloc and start tracking allocations.
        
        This should be called before executing code that needs memory tracking.
        """
        if not self.tracking_enabled:
            tracemalloc.start()
            self.tracking_enabled = True
            self._snapshot_before = tracemalloc.take_snapshot()
    
    def stop_tracking(self) -> None:
        """Stop memory tracking and clean up resources."""
        if self.tracking_enabled:
            tracemalloc.stop()
            self.tracking_enabled = False
            self._snapshot_before = None
    
    def track_allocation(self, obj: Any) -> AllocationInfo:
        """
        Track an object allocation with complete information.
        
        Args:
            obj: The object to track
            
        Returns:
            AllocationInfo containing allocation details including stack trace
        """
        obj_id = id(obj)
        obj_type = type(obj)
        obj_size = sys.getsizeof(obj)
        ref_count = sys.getrefcount(obj) - 1  # Subtract 1 for the parameter reference
        
        # Get traceback if tracking is enabled
        tb = None
        if self.tracking_enabled:
            # Get current snapshot and find this object's allocation
            snapshot = tracemalloc.take_snapshot()
            # Note: tracemalloc tracks by memory address, not object id
            # We'll store None if we can't find the specific traceback
            tb = None
        
        allocation_info = AllocationInfo(
            object_id=obj_id,
            object_type=obj_type,
            size=obj_size,
            traceback=tb,
            timestamp=time.time(),
            reference_count=ref_count
        )
        
        self.allocations[obj_id] = allocation_info
        self.reference_counts[obj_id] = ref_count
        
        return allocation_info
    
    def track_refcount_change(self, obj: Any, delta: int = 0) -> int:
        """
        Track reference count changes for an object.
        
        Args:
            obj: The object to track
            delta: Expected change in reference count (for validation)
            
        Returns:
            Current reference count
        """
        obj_id = id(obj)
        current_refcount = sys.getrefcount(obj) - 1  # Subtract 1 for parameter reference
        
        # Update stored reference count
        old_refcount = self.reference_counts.get(obj_id, 0)
        self.reference_counts[obj_id] = current_refcount
        
        # Update allocation info if it exists
        if obj_id in self.allocations:
            self.allocations[obj_id].reference_count = current_refcount
        
        return current_refcount
    
    def get_memory_layout(self, obj: Any) -> MemoryLayout:
        """
        Generate detailed memory layout information for an object.
        
        Args:
            obj: The object to analyze
            
        Returns:
            MemoryLayout containing comprehensive memory information
        """
        obj_id = id(obj)
        obj_type = type(obj)
        type_name = obj_type.__name__
        size = sys.getsizeof(obj)
        ref_count = sys.getrefcount(obj) - 1  # Subtract 1 for parameter reference
        
        # Extract fields/attributes based on object type
        fields = self._extract_fields(obj)
        
        # Get internal structure for complex types
        internal_structure = self._get_internal_structure(obj)
        
        layout = MemoryLayout(
            object_id=obj_id,
            type_name=type_name,
            size=size,
            reference_count=ref_count,
            fields=fields,
            memory_address=obj_id,  # In CPython, id() returns memory address
            internal_structure=internal_structure
        )
        
        return layout
    
    def _extract_fields(self, obj: Any) -> Dict[str, Any]:
        """
        Extract fields/attributes from an object.
        
        Args:
            obj: The object to extract fields from
            
        Returns:
            Dictionary mapping field names to values
        """
        fields = {}
        
        # Handle different object types
        if isinstance(obj, dict):
            fields = {f"key_{i}": (k, v) for i, (k, v) in enumerate(obj.items())}
        elif isinstance(obj, (list, tuple)):
            fields = {f"item_{i}": v for i, v in enumerate(obj)}
        elif isinstance(obj, set):
            fields = {f"item_{i}": v for i, v in enumerate(obj)}
        elif hasattr(obj, '__dict__'):
            fields = obj.__dict__.copy()
        elif hasattr(obj, '__slots__'):
            fields = {slot: getattr(obj, slot, None) for slot in obj.__slots__}
        else:
            # For primitive types, store the value itself
            fields = {'value': obj}
        
        return fields
    
    def _get_internal_structure(self, obj: Any) -> Optional[Dict[str, Any]]:
        """
        Get internal structure information for complex data types.
        
        Args:
            obj: The object to analyze
            
        Returns:
            Dictionary with internal structure details or None
        """
        structure = {}
        
        if isinstance(obj, list):
            structure['type'] = 'list'
            structure['length'] = len(obj)
            structure['capacity'] = len(obj)  # Python lists don't expose capacity directly
            structure['item_size'] = sys.getsizeof(obj[0]) if obj else 0
            
        elif isinstance(obj, dict):
            structure['type'] = 'dict'
            structure['length'] = len(obj)
            structure['load_factor'] = len(obj) / max(1, len(obj))  # Simplified
            structure['hash_table_size'] = sys.getsizeof(obj)
            
        elif isinstance(obj, tuple):
            structure['type'] = 'tuple'
            structure['length'] = len(obj)
            structure['immutable'] = True
            
        elif isinstance(obj, set):
            structure['type'] = 'set'
            structure['length'] = len(obj)
            structure['hash_table_size'] = sys.getsizeof(obj)
            
        else:
            return None
        
        return structure if structure else None
    
    def get_allocation_info(self, obj: Any) -> Optional[AllocationInfo]:
        """
        Get allocation information for a tracked object.
        
        Args:
            obj: The object to look up
            
        Returns:
            AllocationInfo if object is tracked, None otherwise
        """
        obj_id = id(obj)
        return self.allocations.get(obj_id)
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get overall memory statistics for tracked objects.
        
        Returns:
            Dictionary containing memory statistics
        """
        total_size = sum(info.size for info in self.allocations.values())
        total_objects = len(self.allocations)
        
        # Group by type
        by_type: Dict[str, int] = {}
        for info in self.allocations.values():
            type_name = info.object_type.__name__
            by_type[type_name] = by_type.get(type_name, 0) + 1
        
        stats = {
            'total_size': total_size,
            'total_objects': total_objects,
            'by_type': by_type,
            'tracking_enabled': self.tracking_enabled
        }
        
        return stats
    
    def clear(self) -> None:
        """Clear all tracked allocation data."""
        self.allocations.clear()
        self.reference_counts.clear()
    
    # Memory Layout Visualization Methods
    
    def visualize_memory_layout(self, obj: Any) -> str:
        """
        Create an ASCII diagram visualization of an object's memory layout.
        
        Args:
            obj: The object to visualize
            
        Returns:
            String containing ASCII art representation of memory layout
        """
        layout = self.get_memory_layout(obj)
        
        lines = []
        lines.append("=" * 60)
        lines.append(f"Memory Layout: {layout.type_name}")
        lines.append("=" * 60)
        lines.append(f"Address:    0x{layout.memory_address:016x}")
        lines.append(f"Size:       {layout.size} bytes")
        lines.append(f"Type:       {layout.type_name}")
        lines.append(f"RefCount:   {layout.reference_count}")
        lines.append("-" * 60)
        
        # Add type-specific visualization
        if isinstance(obj, list):
            lines.extend(self._visualize_list(obj, layout))
        elif isinstance(obj, dict):
            lines.extend(self._visualize_dict(obj, layout))
        elif isinstance(obj, tuple):
            lines.extend(self._visualize_tuple(obj, layout))
        elif isinstance(obj, set):
            lines.extend(self._visualize_set(obj, layout))
        else:
            lines.extend(self._visualize_generic(obj, layout))
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def _visualize_list(self, obj: list, layout: MemoryLayout) -> List[str]:
        """Create ASCII visualization for list objects."""
        lines = []
        lines.append("List Structure:")
        lines.append("")
        lines.append("  ┌─────────────────────────────────┐")
        lines.append(f"  │ List Object (len={len(obj)})      │")
        lines.append("  ├─────────────────────────────────┤")
        
        if obj:
            for i, item in enumerate(obj[:5]):  # Show first 5 items
                item_repr = repr(item)[:25]
                lines.append(f"  │ [{i}] -> {item_repr:<25} │")
            
            if len(obj) > 5:
                lines.append(f"  │ ... ({len(obj) - 5} more items)       │")
        else:
            lines.append("  │ (empty)                         │")
        
        lines.append("  └─────────────────────────────────┘")
        
        if layout.internal_structure:
            lines.append("")
            lines.append("Internal Details:")
            lines.append(f"  - Length: {layout.internal_structure.get('length', 0)}")
            lines.append(f"  - Capacity: {layout.internal_structure.get('capacity', 0)}")
        
        return lines
    
    def _visualize_dict(self, obj: dict, layout: MemoryLayout) -> List[str]:
        """Create ASCII visualization for dict objects."""
        lines = []
        lines.append("Dictionary Structure:")
        lines.append("")
        lines.append("  ┌─────────────────────────────────────────┐")
        lines.append(f"  │ Dict Object (len={len(obj)})            │")
        lines.append("  ├─────────────────────────────────────────┤")
        
        if obj:
            items = list(obj.items())[:5]  # Show first 5 items
            for key, value in items:
                key_repr = repr(key)[:15]
                val_repr = repr(value)[:20]
                lines.append(f"  │ {key_repr:<15} : {val_repr:<20} │")
            
            if len(obj) > 5:
                lines.append(f"  │ ... ({len(obj) - 5} more items)           │")
        else:
            lines.append("  │ (empty)                                 │")
        
        lines.append("  └─────────────────────────────────────────┘")
        
        if layout.internal_structure:
            lines.append("")
            lines.append("Internal Details:")
            lines.append(f"  - Entries: {layout.internal_structure.get('length', 0)}")
            lines.append(f"  - Hash Table Size: {layout.internal_structure.get('hash_table_size', 0)} bytes")
        
        return lines
    
    def _visualize_tuple(self, obj: tuple, layout: MemoryLayout) -> List[str]:
        """Create ASCII visualization for tuple objects."""
        lines = []
        lines.append("Tuple Structure (Immutable):")
        lines.append("")
        lines.append("  ┌─────────────────────────────────┐")
        lines.append(f"  │ Tuple Object (len={len(obj)})     │")
        lines.append("  ├─────────────────────────────────┤")
        
        if obj:
            for i, item in enumerate(obj[:5]):  # Show first 5 items
                item_repr = repr(item)[:25]
                lines.append(f"  │ ({i}) -> {item_repr:<25} │")
            
            if len(obj) > 5:
                lines.append(f"  │ ... ({len(obj) - 5} more items)       │")
        else:
            lines.append("  │ (empty)                         │")
        
        lines.append("  └─────────────────────────────────┘")
        
        return lines
    
    def _visualize_set(self, obj: set, layout: MemoryLayout) -> List[str]:
        """Create ASCII visualization for set objects."""
        lines = []
        lines.append("Set Structure:")
        lines.append("")
        lines.append("  ┌─────────────────────────────────┐")
        lines.append(f"  │ Set Object (len={len(obj)})       │")
        lines.append("  ├─────────────────────────────────┤")
        
        if obj:
            items = list(obj)[:5]  # Show first 5 items
            for i, item in enumerate(items):
                item_repr = repr(item)[:25]
                lines.append(f"  │ {{{i}}} -> {item_repr:<25} │")
            
            if len(obj) > 5:
                lines.append(f"  │ ... ({len(obj) - 5} more items)       │")
        else:
            lines.append("  │ (empty)                         │")
        
        lines.append("  └─────────────────────────────────┘")
        
        return lines
    
    def _visualize_generic(self, obj: Any, layout: MemoryLayout) -> List[str]:
        """Create ASCII visualization for generic objects."""
        lines = []
        lines.append("Object Structure:")
        lines.append("")
        
        if layout.fields:
            lines.append("  ┌─────────────────────────────────────────┐")
            lines.append(f"  │ {layout.type_name} Object                │")
            lines.append("  ├─────────────────────────────────────────┤")
            
            for key, value in list(layout.fields.items())[:10]:
                key_repr = str(key)[:15]
                val_repr = repr(value)[:20]
                lines.append(f"  │ {key_repr:<15} : {val_repr:<20} │")
            
            if len(layout.fields) > 10:
                lines.append(f"  │ ... ({len(layout.fields) - 10} more fields)      │")
            
            lines.append("  └─────────────────────────────────────────┘")
        else:
            lines.append(f"  Value: {repr(obj)}")
        
        return lines
    
    def track_mutation(self, obj: Any, operation: str) -> Dict[str, MemoryLayout]:
        """
        Track an object mutation by capturing before and after states.
        
        Args:
            obj: The object being mutated
            operation: Description of the mutation operation
            
        Returns:
            Dictionary with 'before' and 'after' MemoryLayout objects
        """
        # Capture state before mutation
        before_layout = self.get_memory_layout(obj)
        
        # Note: The actual mutation happens outside this method
        # This method should be called before the mutation, then the result
        # should be updated after the mutation
        
        return {
            'before': before_layout,
            'operation': operation,
            'timestamp': time.time()
        }
    
    def complete_mutation_tracking(self, obj: Any, mutation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete mutation tracking by capturing the after state.
        
        Args:
            obj: The object after mutation
            mutation_data: Dictionary returned from track_mutation
            
        Returns:
            Complete mutation tracking data with before and after states
        """
        after_layout = self.get_memory_layout(obj)
        
        mutation_data['after'] = after_layout
        mutation_data['size_change'] = after_layout.size - mutation_data['before'].size
        mutation_data['refcount_change'] = after_layout.reference_count - mutation_data['before'].reference_count
        
        return mutation_data
    
    def visualize_mutation(self, mutation_data: Dict[str, Any]) -> str:
        """
        Create a visualization showing before and after states of a mutation.
        
        Args:
            mutation_data: Dictionary from complete_mutation_tracking
            
        Returns:
            String containing before/after visualization
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"Object Mutation: {mutation_data.get('operation', 'Unknown')}")
        lines.append("=" * 60)
        lines.append("")
        
        before = mutation_data.get('before')
        after = mutation_data.get('after')
        
        if before:
            lines.append("BEFORE:")
            lines.append(f"  Size: {before.size} bytes")
            lines.append(f"  RefCount: {before.reference_count}")
            lines.append(f"  Fields: {len(before.fields)}")
            lines.append("")
        
        if after:
            lines.append("AFTER:")
            lines.append(f"  Size: {after.size} bytes")
            lines.append(f"  RefCount: {after.reference_count}")
            lines.append(f"  Fields: {len(after.fields)}")
            lines.append("")
        
        if 'size_change' in mutation_data:
            size_change = mutation_data['size_change']
            refcount_change = mutation_data['refcount_change']
            lines.append("CHANGES:")
            lines.append(f"  Size: {size_change:+d} bytes")
            lines.append(f"  RefCount: {refcount_change:+d}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)