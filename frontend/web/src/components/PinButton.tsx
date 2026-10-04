import clsx from "clsx";
import { Star } from "lucide-react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { addPin, removePin, type Pin } from "../lib/api";
import { useI18n } from "../i18n";

/** Shared pin mutation with an optimistic update on the ["pins"] query. */
export function usePinToggle() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ pin, pinned }: { pin: Pin; pinned: boolean }) => (pinned ? removePin(pin.asset_type, pin.id) : addPin(pin)),
    onMutate: async ({ pin, pinned }) => {
      await qc.cancelQueries({ queryKey: ["pins"] });
      const prev = qc.getQueryData<{ pins: Pin[] }>(["pins"]);
      const list = prev?.pins ?? [];
      qc.setQueryData(["pins"], {
        pins: pinned ? list.filter((p) => !(p.asset_type === pin.asset_type && p.id === pin.id)) : [...list, pin],
      });
      return { prev };
    },
    onError: (_e, _v, ctx) => ctx?.prev && qc.setQueryData(["pins"], ctx.prev),
    onSettled: () => qc.invalidateQueries({ queryKey: ["pins"] }),
  });
}

export function PinStar({ pin, pinned, className }: { pin: Pin; pinned: boolean; className?: string }) {
  const { t } = useI18n();
  const toggle = usePinToggle();
  return (
    <button
      type="button"
      onClick={(e) => {
        e.stopPropagation();
        e.preventDefault();
        toggle.mutate({ pin, pinned });
      }}
      title={pinned ? t("scout.pin.remove") : t("scout.pin.add")}
      aria-label={pinned ? t("scout.pin.remove") : t("scout.pin.add")}
      aria-pressed={pinned}
      className={clsx(
        "inline-flex size-7 items-center justify-center rounded-xl transition-all hover:bg-hover active:scale-90",
        pinned ? "text-warn" : "text-faint hover:text-bone",
        className,
      )}
    >
      <Star className="size-4" fill={pinned ? "currentColor" : "none"} strokeWidth={1.8} />
    </button>
  );
}
