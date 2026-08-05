import { describe, expect, it } from "vitest";
import { parseParameterValue } from "./IndicatorParameterFields";
import type { IndicatorParameter } from "../types";

const parameter = (kind: IndicatorParameter["kind"]): IndicatorParameter => ({
  name: "value",
  label: "Value",
  kind,
  default: null,
  required: false,
  choices: [],
});

describe("parseParameterValue", () => {
  it("parses calibrated integer tuples", () => {
    expect(parseParameterValue(parameter("integer_tuple"), "10, 20, 9")).toEqual([10, 20, 9]);
  });

  it("preserves literal choice values", () => {
    expect(parseParameterValue(parameter("choice"), "balanced")).toBe("balanced");
  });
});
