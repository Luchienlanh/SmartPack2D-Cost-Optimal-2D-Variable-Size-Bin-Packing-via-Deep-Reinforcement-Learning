def summarize_packing(env, total_items, exec_time=0.0):
    opened_bins = getattr(env, "opened_bins", [])
    if not opened_bins:
        opened_bins = [getattr(env, "bin_types", [{"width": env.width, "height": env.height}])[0]]

    total_piece_area = sum(item[3] * item[4] for item in env.placed_items)
    total_bin_area = sum(bin_cfg["width"] * bin_cfg["height"] for bin_cfg in opened_bins)
    placed_count = len(env.placed_items)

    return {
        "cost": env.total_bin_cost,
        "bins_opened": len(opened_bins),
        "utilization": (total_piece_area / total_bin_area * 100) if total_bin_area > 0 else 0.0,
        "placed_count": placed_count,
        "total_count": total_items,
        "success_rate": (placed_count / total_items) * 100 if total_items else 0.0,
        "time": exec_time,
        "opened_bins": opened_bins,
        "placed_items": env.placed_items,
    }
