import { describe, expect, it } from "vitest"
import {
  rawSourceVectorId,
  sourceRefForPath,
  sourceRefFromRawVectorId,
  sourceStemPathFromRef,
  sourceSummaryPathForRef,
  wikiPageIdFromPath,
} from "./source-identity"

describe("source identity", () => {
  it("preserves source subfolders in canonical refs and summary paths", () => {
    const ref = sourceRefForPath(
      "/proj",
      "/proj/raw/sources/team-a/notes.pdf",
    )

    expect(ref).toBe("raw/sources/team-a/notes.pdf")
    expect(sourceStemPathFromRef(ref)).toBe("team-a/notes")
    expect(sourceSummaryPathForRef(ref)).toBe("wiki/sources/team-a/notes.md")
  })

  it("maps raw source refs to prefixed vector ids and back", () => {
    const id = rawSourceVectorId("raw/sources/team-b/notes.pdf")

    expect(id).toBe("raw-sources/team-b/notes.pdf")
    expect(sourceRefFromRawVectorId(id)).toBe("raw/sources/team-b/notes.pdf")
  })

  it("uses wiki-relative page ids instead of basenames", () => {
    expect(wikiPageIdFromPath("/proj", "/proj/wiki/concepts/notes.md")).toBe(
      "concepts/notes",
    )
    expect(wikiPageIdFromPath("/proj", "/proj/wiki/sources/team-a/notes.md")).toBe(
      "sources/team-a/notes",
    )
  })
})
