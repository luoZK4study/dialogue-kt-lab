#!/bin/bash
# Quick check of all training results
cd /home/user4/dialogue-kt

echo "=== Current time: $(date) ==="
echo ""
echo "=== Training Status ==="
ps aux | grep 'python.*dialogue_kt' | grep -v grep | wc -l
echo "training processes running"
echo ""

echo "=== All kt_method metrics (sorted by AUC) ==="
for f in results/metrics_*.txt; do
    if [ -f "$f" ]; then
        auc=$(grep -oP 'AUC: \K[0-9.]+' "$f" 2>/dev/null | head -1)
        name=$(basename "$f" .txt | sed 's/metrics_//')
        echo "$auc $name"
    fi
done | sort -rn | while read auc name; do
    printf "  %6s  %s\n" "$auc" "$name"
done
echo ""

echo "=== Recently modified metrics (last 2 hours) ==="
find results/ -name 'metrics_*.txt' -mmin -120 -ls 2>/dev/null | head -20
echo ""

echo "=== Wave 1 progress ==="
for log in results/training_logs/batch_gpu*.log; do
    if [ -f "$log" ]; then
        last_line=$(grep -E '(=== Training|>>> .* AUC|===.*DONE)' "$log" | tail -5)
        echo "--- $(basename $log) ---"
        echo "$last_line"
    fi
done
