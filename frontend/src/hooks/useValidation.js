/**
 * DepthWizard — useValidation hook (Phase 12)
 *
 * Encapsulates fetching and state management for:
 * - Reference DEM metadata: GET /scenes/{id}/reference
 * - Validation metrics (RMSE, MAE, Correlation): GET /scenes/{id}/validation
 * - Error map metadata and bounds: GET /scenes/{id}/validation/error-map
 *
 * Spec §16, §17, §62, §63, §64.
 */
import { useState, useEffect, useCallback } from 'react';
import { getValidation, getErrorMap } from '../api/validation.js';
import { getReference } from '../api/results.js';

/**
 * @param {string|null} sceneId
 * @param {boolean} [isGeoreferenced=true]
 */
export function useValidation(sceneId, isGeoreferenced = true) {
  const [reference, setReference] = useState(null);
  const [validation, setValidation] = useState(null);
  const [errorMap, setErrorMap]   = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError]         = useState(null);

  const fetchValidationData = useCallback(async () => {
    if (!sceneId) return;

    // Non-georeferenced scenes cannot have reference elevation
    if (!isGeoreferenced) {
      setReference({ available: false, reason: 'Non-georeferenced scene (relative elevation mode)' });
      setValidation({ available: false, metrics: null, reason: 'Validation requires georeferenced imagery' });
      setErrorMap(null);
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const [refRes, valRes, errRes] = await Promise.allSettled([
        getReference(sceneId),
        getValidation(sceneId),
        getErrorMap(sceneId),
      ]);

      if (refRes.status === 'fulfilled') {
        setReference(refRes.value ?? { available: false });
      } else {
        setReference({ available: false, reason: 'No reference elevation data available for this scene.' });
      }

      if (valRes.status === 'fulfilled') {
        setValidation(valRes.value ?? { available: false });
      } else {
        setValidation({ available: false, metrics: null });
      }

      if (errRes.status === 'fulfilled') {
        setErrorMap(errRes.value ?? null);
      } else {
        setErrorMap(null);
      }
    } catch (err) {
      setError(err);
    } finally {
      setIsLoading(false);
    }
  }, [sceneId, isGeoreferenced]);

  useEffect(() => {
    fetchValidationData();
  }, [fetchValidationData]);

  return {
    reference,
    validation,
    errorMap,
    isLoading,
    error,
    refetch: fetchValidationData,
  };
}
